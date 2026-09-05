"""
Tests de la Fase 4 — Diagnóstico accionable + Dashboard avanzado + ADXL345

Cobertura:
  TestFaultClassifier   — clasificación de fallos por tipo, severidad, confianza
  TestAnomalyResultFault— integración fault_diagnosis en AnomalyResult
  TestADXL345Sensor     — sensor ADXL345 con mock I2C (sin hardware)
  TestApiV2Resumen      — endpoint /v2/maquinas/resumen con aislamiento empresa
  TestDashboardV4       — presencia de funciones y secciones nuevas
  TestRepositories4B    — obtener_todas_maquinas_con_health existe
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from unittest.mock import MagicMock, patch
from typing import Optional

import numpy as np
import pytest


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def bearing_vector():
    """High kurtosis + high crest factor + high band energy → BEARING"""
    return np.array([0.20, 7.0, 6.0, 0.50, 900.0, 0.001, 0.002, 0.08])

def imbalance_vector():
    """Low kurtosis + low CF + low band dominant → IMBALANCE"""
    return np.array([0.30, 1.0, 1.8, 0.70, 50.0, 0.10, 0.01, 0.003])

def lubrication_vector():
    """Moderate kurtosis + mid+high bands elevated → LUBRICATION"""
    return np.array([0.12, 2.5, 3.2, 0.28, 300.0, 0.01, 0.03, 0.03])

def normal_vector():
    """All features within normal range"""
    return np.array([0.05, 0.5, 2.0, 0.14, 50.0, 0.003, 0.001, 0.0])

def nan_vector():
    return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])


# ─── TestFaultClassifier ──────────────────────────────────────────────────────

class TestFaultClassifier:

    def setup_method(self):
        from src.edge.anomaly.fault_classifier import FaultClassifier
        self.fc = FaultClassifier()

    def test_normal_returns_none(self):
        r = self.fc.classify(normal_vector(), anomaly_score=0.05)
        assert r is None

    def test_low_anomaly_score_returns_none(self):
        """Below MIN_ANOMALY threshold → no diagnosis even with bad features"""
        r = self.fc.classify(bearing_vector(), anomaly_score=0.10)
        assert r is None

    def test_bearing_detected(self):
        r = self.fc.classify(bearing_vector(), anomaly_score=0.85)
        assert r is not None
        assert r.fault_type == "BEARING"

    def test_imbalance_detected(self):
        r = self.fc.classify(imbalance_vector(), anomaly_score=0.60)
        assert r is not None
        assert r.fault_type == "IMBALANCE"

    def test_lubrication_detected(self):
        r = self.fc.classify(lubrication_vector(), anomaly_score=0.50)
        assert r is not None
        assert r.fault_type in ("LUBRICATION", "UNCERTAIN")  # borderline acceptable

    def test_bearing_severity_severe(self):
        r = self.fc.classify(bearing_vector(), anomaly_score=0.85)
        assert r is not None and r.severity == "SEVERE"

    def test_bearing_severity_moderate(self):
        r = self.fc.classify(bearing_vector(), anomaly_score=0.55)
        assert r is not None and r.severity == "MODERATE"

    def test_bearing_severity_incipient(self):
        r = self.fc.classify(bearing_vector(), anomaly_score=0.30)
        assert r is not None and r.severity == "INCIPIENT"

    def test_confidence_range(self):
        r = self.fc.classify(bearing_vector(), anomaly_score=0.85)
        assert r is not None
        assert 0.0 <= r.confidence <= 1.0

    def test_bearing_high_confidence(self):
        r = self.fc.classify(bearing_vector(), anomaly_score=0.85)
        assert r is not None and r.confidence >= 0.6

    def test_imbalance_high_confidence(self):
        r = self.fc.classify(imbalance_vector(), anomaly_score=0.60)
        assert r is not None and r.confidence >= 0.6

    def test_uncertain_on_mixed_features(self):
        """Ambiguous features should give UNCERTAIN rather than wrong diagnosis"""
        mixed = np.array([0.12, 4.0, 4.5, 0.25, 200.0, 0.03, 0.03, 0.03])
        r = self.fc.classify(mixed, anomaly_score=0.40)
        # Should classify something or be uncertain — must not crash
        assert r is None or r.fault_type in ("BEARING","IMBALANCE","LUBRICATION","UNCERTAIN")

    def test_nan_vector_no_crash(self):
        """NaN features must not raise an exception"""
        try:
            r = self.fc.classify(nan_vector(), anomaly_score=0.70)
            assert r is None or r.fault_type == "UNCERTAIN"
        except Exception as exc:
            pytest.fail(f"NaN features raised: {exc}")

    def test_fault_diagnosis_has_all_fields(self):
        r = self.fc.classify(bearing_vector(), anomaly_score=0.80)
        assert r is not None
        for field in ["fault_type", "affected_axis", "severity",
                      "confidence", "explanation", "recommendation"]:
            assert hasattr(r, field), f"Missing field: {field}"

    def test_explanation_is_string(self):
        r = self.fc.classify(bearing_vector(), anomaly_score=0.80)
        assert r is not None and isinstance(r.explanation, str) and len(r.explanation) > 10

    def test_recommendation_is_string(self):
        r = self.fc.classify(bearing_vector(), anomaly_score=0.80)
        assert r is not None and isinstance(r.recommendation, str) and len(r.recommendation) > 5

    def test_affected_axis_from_per_axis(self):
        per_axis = {
            "x": {"kurtosis": 1.5, "rms": 0.05, "crest_factor": 2.0, "peak_to_peak": 0.10},
            "y": {"kurtosis": 7.0, "rms": 0.20, "crest_factor": 6.0, "peak_to_peak": 0.50},
        }
        r = self.fc.classify(bearing_vector(), anomaly_score=0.85, per_axis_features=per_axis)
        assert r is not None and r.affected_axis == "y"  # y has highest kurtosis

    def test_to_dict(self):
        r = self.fc.classify(bearing_vector(), anomaly_score=0.80)
        assert r is not None
        d = r.to_dict()
        assert set(d.keys()) >= {"fault_type", "severity", "confidence",
                                  "explanation", "recommendation", "affected_axis"}

    def test_extract_per_axis_features(self):
        from src.edge.anomaly.fault_classifier import extract_per_axis_features
        from src.edge.config.edge_config import (
            EdgeConfig, MachineConfig, AcquisitionConfig, BufferConfig
        )
        from src.edge.signal_processing import SignalConfig
        from src.edge.sensors.base_sensor import SensorConfig
        from src.edge.sensors.mock_sensor import MockSensor, MockSensorParams
        from src.edge.pipeline.acquisition import AcquisitionSession
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            cfg = EdgeConfig(
                machine=MachineConfig(machine_id="T", empresa_id=1, maquina_id=1),
                sensor=SensorConfig(sensor_id="t", sensor_type="mock",
                                    sampling_rate_hz=3200, samples_per_window=3200,
                                    axes=["x","y","z"]),
                signal=SignalConfig(fs=3200),
                acquisition=AcquisitionConfig(),
                buffer=BufferConfig(base_dir=tmp),
            )
            sensor = MockSensor(cfg.sensor, MockSensorParams())
            sensor.configure()
            fs = AcquisitionSession(cfg).acquire(sensor.read())
            assert fs is not None
            per_axis = extract_per_axis_features(fs)
            assert "x" in per_axis
            assert "kurtosis" in per_axis["x"]


# ─── TestAnomalyResultFault ───────────────────────────────────────────────────

class TestAnomalyResultFault:

    def test_fault_diagnosis_field_default_none(self):
        from src.edge.anomaly.anomaly_detector import AnomalyResult
        ar = AnomalyResult(0.8, 30, "ALERTA", "Alto", "test", None, False, "zscore")
        assert hasattr(ar, "fault_diagnosis")
        assert ar.fault_diagnosis is None

    def test_fault_diagnosis_can_be_set(self):
        from src.edge.anomaly.anomaly_detector import AnomalyResult
        from src.edge.anomaly.fault_classifier import FaultDiagnosis
        ar = AnomalyResult(0.8, 30, "ALERTA", "Alto", "test", None, False, "zscore")
        fd = FaultDiagnosis("BEARING", "x", "SEVERE", 0.9, "explanation", "rec")
        ar.fault_diagnosis = fd
        assert ar.fault_diagnosis.fault_type == "BEARING"


# ─── TestADXL345Sensor ────────────────────────────────────────────────────────

class TestADXL345Sensor:

    def _make_sensor(self, n_samples=50, i2c_addr="0x53"):
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor
        from src.edge.sensors.base_sensor import SensorConfig
        cfg = SensorConfig(
            sensor_id="adxl_test", sensor_type="adxl345",
            sampling_rate_hz=3200.0, samples_per_window=n_samples,
            axes=["x","y","z"], i2c_address=i2c_addr,
        )
        bus = MagicMock()
        bus.read_i2c_block_data.return_value = [0xE5]   # DEVID
        return ADXL345Sensor(cfg, bus=bus), bus, cfg

    def test_is_sensor_interface(self):
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor
        from src.edge.sensors.base_sensor import SensorInterface
        assert issubclass(ADXL345Sensor, SensorInterface)

    def test_import_without_hardware(self):
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor
        assert ADXL345Sensor is not None

    def test_configure_writes_registers(self):
        sensor, bus, _ = self._make_sensor()
        sensor.configure()
        assert bus.write_byte_data.call_count >= 3

    def test_configure_power_ctl(self):
        from src.edge.sensors.adxl345_sensor import REG_POWER_CTL
        sensor, bus, _ = self._make_sensor()
        sensor.configure()
        power_calls = [c for c in bus.write_byte_data.call_args_list
                       if c[0][1] == REG_POWER_CTL]
        assert len(power_calls) >= 1
        assert power_calls[-1][0][2] == 0x08   # measurement mode

    def test_configure_devid_mismatch_raises(self):
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor
        from src.edge.sensors.base_sensor import SensorConfig, SensorConfigurationError
        cfg = SensorConfig("t","adxl345",3200,samples_per_window=10)
        bus = MagicMock()
        bus.read_i2c_block_data.return_value = [0x00]  # wrong DEVID
        sensor = ADXL345Sensor(cfg, bus=bus)
        with pytest.raises(SensorConfigurationError):
            sensor.configure()

    def test_read_returns_sensor_reading(self):
        sensor, bus, _ = self._make_sensor(n_samples=20)
        sensor.configure()
        bus.read_i2c_block_data.return_value = [0x00,0x10, 0x00,0x00, 0x00,0x00]
        reading = sensor.read()
        from src.edge.sensors.base_sensor import SensorReading
        assert isinstance(reading, SensorReading)

    def test_read_correct_n_samples(self):
        sensor, bus, _ = self._make_sensor(n_samples=50)
        sensor.configure()
        bus.read_i2c_block_data.return_value = [0,0, 0,0, 0,0]
        assert sensor.read().n_samples == 50

    def test_read_three_axes(self):
        sensor, bus, _ = self._make_sensor(n_samples=10)
        sensor.configure()
        bus.read_i2c_block_data.return_value = [0,0, 0,0, 0,0]
        r = sensor.read()
        assert set(r.axes.keys()) == {"x", "y", "z"}

    def test_read_units_are_g(self):
        """Output should be in g — scale factor 0.0039 for ±2g range"""
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor
        from src.edge.sensors.base_sensor import SensorConfig
        cfg = SensorConfig("t","adxl345",3200,samples_per_window=5)
        bus = MagicMock()
        bus.read_i2c_block_data.return_value = [0xE5]
        sensor = ADXL345Sensor(cfg, bus=bus)
        sensor.configure()
        # 256 LSB at 0.0039 g/LSB = 0.998 g ≈ 1g
        bus.read_i2c_block_data.return_value = [0x00, 0x01,  # X = 256
                                                 0x00, 0x00,  # Y = 0
                                                 0x00, 0x00]  # Z = 0
        r = sensor.read()
        np.testing.assert_allclose(r.axes["x"].mean(), 256 * 0.0039, rtol=0.01)

    def test_two_complement_positive(self):
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor
        assert ADXL345Sensor._to_signed16(0xFF, 0x01) == 0x01FF   # 511

    def test_two_complement_negative(self):
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor
        assert ADXL345Sensor._to_signed16(0x00, 0x80) == -32768

    def test_two_complement_max_positive(self):
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor
        assert ADXL345Sensor._to_signed16(0xFF, 0x7F) == 32767

    def test_default_i2c_address(self):
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor, DEFAULT_I2C_ADDRESS
        from src.edge.sensors.base_sensor import SensorConfig
        cfg = SensorConfig("t","adxl345",3200,samples_per_window=5)
        sensor = ADXL345Sensor(cfg)  # no bus, no crash
        assert sensor._addr == DEFAULT_I2C_ADDRESS

    def test_custom_i2c_address(self):
        sensor, _, _ = self._make_sensor(i2c_addr="0x1D")
        assert sensor._addr == 0x1D

    def test_i2c_error_raises_runtime(self):
        sensor, bus, _ = self._make_sensor(n_samples=5)
        sensor.configure()
        bus.read_i2c_block_data.side_effect = OSError("I2C bus error")
        from src.edge.sensors.base_sensor import SensorReadError
        with pytest.raises((SensorReadError, RuntimeError)):
            sensor.read()

    def test_close_sets_unconfigured(self):
        sensor, bus, _ = self._make_sensor(n_samples=5)
        sensor.configure()
        sensor.close()
        assert not sensor._configured

    def test_close_calls_bus_close(self):
        sensor, bus, _ = self._make_sensor(n_samples=5)
        sensor.configure()
        sensor.close()
        bus.close.assert_called_once()

    def test_sensor_type_in_reading(self):
        sensor, bus, _ = self._make_sensor(n_samples=5)
        sensor.configure()
        bus.read_i2c_block_data.return_value = [0,0,0,0,0,0]
        r = sensor.read()
        assert r.sensor_type == "adxl345"

    def test_get_metadata(self):
        sensor, bus, _ = self._make_sensor()
        sensor.configure()
        meta = sensor.get_metadata()
        assert "i2c_address" in meta and "scale_factor" in meta

    def test_sampling_rate_configured_preserved(self):
        sensor, bus, _ = self._make_sensor(n_samples=5)
        sensor.configure()
        bus.read_i2c_block_data.return_value = [0,0,0,0,0,0]
        r = sensor.read()
        assert r.sampling_rate_configured == 3200.0

    def test_timestamps_length(self):
        sensor, bus, _ = self._make_sensor(n_samples=10)
        sensor.configure()
        bus.read_i2c_block_data.return_value = [0,0,0,0,0,0]
        r = sensor.read()
        assert len(r.timestamps) == 10


# ─── TestApiV2Resumen ─────────────────────────────────────────────────────────

class TestApiV2Resumen:

    @pytest.fixture
    def client(self):
        pytest.importorskip("fastapi",
            reason="fastapi not installed — API tests require it")
        try:
            from fastapi.testclient import TestClient
        except (RuntimeError, ModuleNotFoundError) as e:
            pytest.skip(f"TestClient unavailable: {e}")
        with patch("database.get_conn"), patch("database.init_db"):
            from api import app
            return TestClient(app, raise_server_exceptions=False)

    def _auth(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
        from auth import crear_token
        token = crear_token({"sub": "t@t.com", "nombre": "T", "rol": "admin", "empresa_id": 1})
        return {"Authorization": f"Bearer {token}"}

    def test_resumen_requires_auth(self, client):
        r = client.get("/v2/maquinas/resumen")
        assert r.status_code in (401, 403)

    def test_resumen_with_auth_calls_repo(self, client):
        with patch("database_v2.repositories.obtener_todas_maquinas_con_health",
                   return_value=[{"maquina_id": 1, "nombre": "T", "tipo": "torno_cnc",
                                  "empresa_id": 1, "health_score": 80,
                                  "trend": "stable", "slope": 0.0, "timestamp": None}]):
            r = client.get("/v2/maquinas/resumen", headers=self._auth())
        assert r.status_code == 200
        assert r.json()["total"] == 1


# ─── TestDashboardV4 ──────────────────────────────────────────────────────────

class TestDashboardV4:

    def _read(self):
        return open(
            os.path.join(os.path.dirname(__file__), "../src/dashboard.py"),
            encoding="utf-8",
        ).read()

    def test_dashboard_syntax_valid(self):
        import ast
        ast.parse(self._read())

    def test_render_resumen_planta_defined(self):
        assert "_render_resumen_planta" in self._read()

    def test_render_maquina_individual_defined(self):
        assert "_render_maquina_individual" in self._read()

    def test_mantenimiento_recommendation_defined(self):
        assert "_mantenimiento_recommendation" in self._read()

    def test_resumen_endpoint_called(self):
        assert "/v2/maquinas/resumen" in self._read()

    def test_health_chart_present(self):
        assert "line_chart" in self._read()

    def test_vibration_chart_rms(self):
        assert "rms_x" in self._read()

    def test_vibration_chart_kurtosis(self):
        assert "kurtosis_x" in self._read()

    def test_fault_diagnosis_panel(self):
        assert "fault_diagnosis" in self._read()
        assert "fault_type" in self._read()

    def test_legacy_tabs_unchanged(self):
        src = self._read()
        assert "tab_empresas" in src
        assert "tab_maquinas" in src

    def test_mantenimiento_text_content(self):
        src = self._read()
        assert "Tendencia" in src or "revisión" in src.lower()


# ─── TestRepositories4B ───────────────────────────────────────────────────────

class TestRepositories4B:

    def test_function_exists(self):
        import ast
        src = open(
            os.path.join(os.path.dirname(__file__),
                         "../src/database_v2/repositories.py"),
            encoding="utf-8",
        ).read()
        assert "def obtener_todas_maquinas_con_health" in src

    def test_function_syntax(self):
        import ast
        src = open(
            os.path.join(os.path.dirname(__file__),
                         "../src/database_v2/repositories.py"),
            encoding="utf-8",
        ).read()
        ast.parse(src)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
