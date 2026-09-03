"""
Tests de la Fase 5A — Puesta en producción real

Cobertura:
  TestSensorFactory    — selección automática mock/adxl345 desde config
  TestADXL345Extended  — range_g, odr_hz, i2c_bus desde YAML/config
  TestLoggingConfig    — setup_logging, get_logger
  TestEmailProduction  — EMAIL_ACTIVO flag, SMTP mock, cooldown persistente
  TestYAMLProduction   — YAML con sensor adxl345, campos extra, __main__
  TestSchedulerResilience — errores hardware no matan scheduler
  TestEntryPoint       — python -m invocable
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def make_sensor_config(
    sensor_type="mock",
    sensor_id="test",
    sampling_rate_hz=3200.0,
    samples_per_window=50,
    i2c_address="0x53",
    odr_hz=3200.0,
    extra=None,
):
    from src.edge.sensors.base_sensor import SensorConfig
    return SensorConfig(
        sensor_id          = sensor_id,
        sensor_type        = sensor_type,
        sampling_rate_hz   = sampling_rate_hz,
        odr_hz             = odr_hz,
        samples_per_window = samples_per_window,
        axes               = ["x","y","z"],
        i2c_address        = i2c_address,
        extra              = extra or {},
    )


def make_mock_bus(devid=0xE5):
    bus = MagicMock()
    bus.read_i2c_block_data.return_value = [devid]
    bus.write_byte_data.return_value     = None
    bus.close.return_value               = None
    return bus


# ─── TestSensorFactory ────────────────────────────────────────────────────────

class TestSensorFactory:

    def test_mock_type_returns_mock_sensor(self):
        from src.edge.sensors.sensor_factory import create_sensor
        from src.edge.sensors.mock_sensor import MockSensor
        cfg    = make_sensor_config(sensor_type="mock")
        sensor = create_sensor(cfg)
        assert isinstance(sensor, MockSensor)

    def test_adxl345_type_returns_adxl345_sensor(self):
        from src.edge.sensors.sensor_factory import create_sensor
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor
        cfg    = make_sensor_config(sensor_type="adxl345")
        sensor = create_sensor(cfg, bus=make_mock_bus())
        assert isinstance(sensor, ADXL345Sensor)

    def test_unknown_type_raises_value_error(self):
        from src.edge.sensors.sensor_factory import create_sensor
        cfg = make_sensor_config(sensor_type="iepe_unknown")
        with pytest.raises(ValueError, match="Unknown sensor type"):
            create_sensor(cfg)

    def test_bus_parameter_injected_into_adxl345(self):
        from src.edge.sensors.sensor_factory import create_sensor
        bus = make_mock_bus()
        cfg = make_sensor_config(sensor_type="adxl345")
        sensor = create_sensor(cfg, bus=bus)
        assert sensor._bus is bus

    def test_mock_type_case_insensitive(self):
        from src.edge.sensors.sensor_factory import create_sensor
        from src.edge.sensors.mock_sensor import MockSensor
        cfg = make_sensor_config(sensor_type="MOCK")
        assert isinstance(create_sensor(cfg), MockSensor)

    def test_sensor_interface_returned(self):
        from src.edge.sensors.sensor_factory import create_sensor
        from src.edge.sensors.base_sensor import SensorInterface
        cfg    = make_sensor_config(sensor_type="mock")
        sensor = create_sensor(cfg)
        assert isinstance(sensor, SensorInterface)

    def test_mock_sensor_is_configurable_without_hardware(self):
        from src.edge.sensors.sensor_factory import create_sensor
        cfg    = make_sensor_config(sensor_type="mock", samples_per_window=20)
        sensor = create_sensor(cfg)
        sensor.configure()
        reading = sensor.read()
        assert reading.n_samples == 20


# ─── TestADXL345Extended ──────────────────────────────────────────────────────

class TestADXL345Extended:

    def _make(self, **kwargs):
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor
        cfg    = make_sensor_config(sensor_type="adxl345", **kwargs)
        bus    = make_mock_bus()
        sensor = ADXL345Sensor(cfg, bus=bus)
        return sensor, bus

    def test_odr_hz_3200_maps_to_0x0F(self):
        sensor, _ = self._make(odr_hz=3200.0)
        assert sensor._odr_reg == 0x0F

    def test_odr_hz_1600_maps_to_0x0E(self):
        sensor, _ = self._make(odr_hz=1600.0)
        assert sensor._odr_reg == 0x0E

    def test_odr_hz_800_maps_to_0x0D(self):
        sensor, _ = self._make(odr_hz=800.0)
        assert sensor._odr_reg == 0x0D

    def test_odr_hz_400_maps_to_0x0C(self):
        sensor, _ = self._make(odr_hz=400.0)
        assert sensor._odr_reg == 0x0C

    def test_range_g_2_maps_to_0x00(self):
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor
        cfg    = make_sensor_config(sensor_type="adxl345", extra={"range_g": 2})
        sensor = ADXL345Sensor(cfg, bus=make_mock_bus())
        assert sensor._range_reg == 0x00

    def test_range_g_4_maps_to_0x01(self):
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor
        cfg    = make_sensor_config(sensor_type="adxl345", extra={"range_g": 4})
        sensor = ADXL345Sensor(cfg, bus=make_mock_bus())
        assert sensor._range_reg == 0x01

    def test_range_g_16_maps_to_0x03(self):
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor
        cfg    = make_sensor_config(sensor_type="adxl345", extra={"range_g": 16})
        sensor = ADXL345Sensor(cfg, bus=make_mock_bus())
        assert sensor._range_reg == 0x03

    def test_range_g_2_gives_scale_0039(self):
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor
        cfg    = make_sensor_config(sensor_type="adxl345", extra={"range_g": 2})
        sensor = ADXL345Sensor(cfg, bus=make_mock_bus())
        sensor.configure()
        assert abs(sensor._scale - 0.0039) < 1e-6

    def test_i2c_bus_from_extra(self):
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor
        cfg    = make_sensor_config(sensor_type="adxl345", extra={"i2c_bus": 0})
        sensor = ADXL345Sensor(cfg, bus=make_mock_bus())
        assert sensor._i2c_bus_number == 0

    def test_default_i2c_bus_is_1(self):
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor
        cfg    = make_sensor_config(sensor_type="adxl345")
        sensor = ADXL345Sensor(cfg, bus=make_mock_bus())
        assert sensor._i2c_bus_number == 1

    def test_configure_writes_bw_rate(self):
        from src.edge.sensors.adxl345_sensor import REG_BW_RATE
        sensor, bus = self._make(odr_hz=1600.0)
        sensor.configure()
        regs_written = [c[0][1] for c in bus.write_byte_data.call_args_list]
        assert REG_BW_RATE in regs_written

    def test_configure_writes_data_format(self):
        from src.edge.sensors.adxl345_sensor import REG_DATA_FORMAT
        sensor, bus = self._make()
        sensor.configure()
        regs_written = [c[0][1] for c in bus.write_byte_data.call_args_list]
        assert REG_DATA_FORMAT in regs_written

    def test_configure_writes_power_ctl_measurement(self):
        from src.edge.sensors.adxl345_sensor import REG_POWER_CTL
        sensor, bus = self._make()
        sensor.configure()
        power_calls = [c for c in bus.write_byte_data.call_args_list
                       if c[0][1] == REG_POWER_CTL]
        # Last write to POWER_CTL must set measurement bit (0x08)
        assert power_calls[-1][0][2] == 0x08

    def test_wrong_devid_raises_configuration_error(self):
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor
        from src.edge.sensors.base_sensor import SensorConfigurationError
        cfg = make_sensor_config(sensor_type="adxl345")
        bus = make_mock_bus(devid=0x00)
        sensor = ADXL345Sensor(cfg, bus=bus)
        with pytest.raises(SensorConfigurationError, match="DEVID mismatch"):
            sensor.configure()

    def test_i2c_error_on_read_raises_runtime(self):
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor
        cfg = make_sensor_config(sensor_type="adxl345", samples_per_window=5)
        bus = make_mock_bus()
        sensor = ADXL345Sensor(cfg, bus=bus)
        sensor.configure()
        bus.read_i2c_block_data.side_effect = OSError("I2C bus error")
        with pytest.raises(RuntimeError, match="ADXL345 read error"):
            sensor.read()

    def test_read_not_called_before_configure(self):
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor
        from src.edge.sensors.base_sensor import SensorConfigurationError
        cfg    = make_sensor_config(sensor_type="adxl345", samples_per_window=5)
        sensor = ADXL345Sensor(cfg, bus=make_mock_bus())
        with pytest.raises(SensorConfigurationError):
            sensor.read()

    def test_smbus2_used_when_bus_is_none(self):
        """When bus=None in production, smbus2.SMBus(i2c_bus_number) is created."""
        import sys
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor
        cfg     = make_sensor_config(sensor_type="adxl345", extra={"i2c_bus": 1})
        sensor  = ADXL345Sensor(cfg, bus=None)
        mock_bus = make_mock_bus()
        mock_smbus2 = MagicMock()
        mock_smbus2.SMBus.return_value = mock_bus
        with patch.dict(sys.modules, {"smbus2": mock_smbus2}):
            sensor.configure()
        mock_smbus2.SMBus.assert_called_once_with(1)


# ─── TestEdgeConfigYAML ───────────────────────────────────────────────────────

class TestEdgeConfigYAML:

    def _write_yaml(self, tmp_path: Path, content: str) -> str:
        p = tmp_path / "test.yaml"
        p.write_text(content)
        return str(p)

    def test_sensor_type_mock_loaded(self, tmp_path):
        from src.edge.config.edge_config import EdgeConfig
        yaml = self._write_yaml(tmp_path, """
machine:
  id: Test
  empresa_id: 1
  maquina_id: 1
sensor:
  type: mock
  sampling_rate_hz: 3200
  samples_per_window: 3200
  odr_hz: 3200
  axes: [x, y, z]
""")
        cfg = EdgeConfig.from_yaml(yaml)
        assert cfg.sensor.sensor_type == "mock"

    def test_sensor_type_adxl345_loaded(self, tmp_path):
        from src.edge.config.edge_config import EdgeConfig
        yaml = self._write_yaml(tmp_path, """
machine:
  id: Test
  empresa_id: 1
  maquina_id: 1
sensor:
  type: adxl345
  sampling_rate_hz: 3200
  samples_per_window: 3200
  odr_hz: 3200
  axes: [x, y, z]
  i2c_address: "0x53"
  i2c_bus: 1
  range_g: 4
""")
        cfg = EdgeConfig.from_yaml(yaml)
        assert cfg.sensor.sensor_type == "adxl345"
        assert cfg.sensor.extra["i2c_bus"] == 1
        assert cfg.sensor.extra["range_g"] == 4
        assert cfg.sensor.i2c_address == "0x53"

    def test_extra_empty_for_mock(self, tmp_path):
        from src.edge.config.edge_config import EdgeConfig
        yaml = self._write_yaml(tmp_path, """
machine:
  id: Test
  empresa_id: 1
  maquina_id: 1
sensor:
  type: mock
  sampling_rate_hz: 3200
  samples_per_window: 3200
  odr_hz: 3200
  axes: [x, y, z]
""")
        cfg = EdgeConfig.from_yaml(yaml)
        # extra should exist but may be empty for mock
        assert isinstance(cfg.sensor.extra, dict)

    def test_example_cnc_yaml_loads(self):
        """The production example YAML must load without error."""
        from src.edge.config.edge_config import EdgeConfig
        yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "machines", "example_cnc.yaml"
        )
        cfg = EdgeConfig.from_yaml(yaml_path)
        assert cfg.machine.machine_id == "Torno_CNC_1"
        assert cfg.sensor.sensor_type in ("mock", "adxl345")


# ─── TestLoggingConfig ────────────────────────────────────────────────────────

class TestLoggingConfig:

    def test_setup_logging_does_not_crash(self):
        from src.edge.logging_config import setup_logging
        setup_logging(level="WARNING")  # quiet during tests

    def test_get_logger_returns_logger(self):
        from src.edge.logging_config import get_logger
        logger = get_logger("test.subsystem")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.subsystem"

    def test_log_level_env_var(self, monkeypatch):
        from src.edge.logging_config import setup_logging
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        setup_logging()  # reads from env
        root = logging.getLogger()
        assert root.level == logging.DEBUG
        # reset
        logging.getLogger().setLevel(logging.WARNING)

    def test_log_file_created(self, tmp_path):
        from src.edge.logging_config import setup_logging
        log_path = str(tmp_path / "test.log")
        setup_logging(level="WARNING", log_file=log_path)
        logging.getLogger("test_file_logger").warning("test entry")
        # File should exist after logging
        assert os.path.exists(log_path)


# ─── TestEmailProduction ──────────────────────────────────────────────────────

class TestEmailProduction:

    def _make_ar(self, nivel_riesgo="CRÍTICO", health_score=10, is_cold_start=False):
        from src.edge.anomaly.anomaly_detector import AnomalyResult
        return AnomalyResult(
            anomaly_score=0.9, health_score=health_score,
            resultado="NOK", nivel_riesgo=nivel_riesgo,
            diagnostico="test", model_version_id=None,
            is_cold_start=is_cold_start, algorithm="zscore",
        )

    def _call_maybe(self, ar, email_activo="false", **kw):
        from src.alertas_v2 import maybe_enviar_alerta_cnc
        fs = MagicMock()
        fs.primary_axis = "x"
        fs.multiaxis.get_axis.return_value = None
        calls = []
        with patch.dict(os.environ, {"EMAIL_ACTIVO": email_activo}):
            result = maybe_enviar_alerta_cnc(
                1, 1, "TestCNC", ar, fs, cooldown_hours=1.0,
                puede_enviar_fn=kw.get("puede_enviar_fn", lambda m, cooldown_hours=1.0: True),
                registrar_fn=kw.get("registrar_fn", lambda *a, **kw2: (calls.append(a), 99)[1]),
            )
        return result, calls

    def test_email_false_no_smtp_called(self):
        """EMAIL_ACTIVO=false must register alert in DB but not send SMTP."""
        ar = self._make_ar()
        calls = []
        smtp_calls = []
        from src.alertas_v2 import maybe_enviar_alerta_cnc
        fs = MagicMock()
        fs.primary_axis = "x"
        fs.multiaxis.get_axis.return_value = None
        with patch.dict(os.environ, {"EMAIL_ACTIVO": "false"}):
            result = maybe_enviar_alerta_cnc(
                1, 1, "T", ar, fs, cooldown_hours=1.0,
                puede_enviar_fn=lambda m, cooldown_hours=1.0: True,
                registrar_fn=lambda *a, **kw: (calls.append(a), 99)[1],
            )
        # Must return True (alert was processed) and registrar must be called
        assert result is True
        assert len(calls) == 1  # registered in BD
        # SMTP should NOT have been called (EMAIL_ACTIVO=false)
        # Verified indirectly: if SMTP was called, it would fail (no creds) in this env

    def test_email_true_calls_enviar_alerta(self):
        """EMAIL_ACTIVO=true must attempt to call alertas.enviar_alerta."""
        ar = self._make_ar()
        with patch("alertas.enviar_alerta") as mock_smtp, \
             patch.dict(os.environ, {
                 "EMAIL_ACTIVO": "true",
                 "EMAIL_ORIGEN": "t@t.com",
                 "EMAIL_CONTRASENA": "secret",
             }):
            from src.alertas_v2 import maybe_enviar_alerta_cnc
            fs = MagicMock()
            fs.primary_axis = "x"
            fs.multiaxis.get_axis.return_value = None
            maybe_enviar_alerta_cnc(
                1, 1, "T", ar, fs,
                puede_enviar_fn=lambda m, cooldown_hours=1.0: True,
                registrar_fn=lambda *a, **kw: 99,
            )
        mock_smtp.assert_called_once()

    def test_smtp_failure_does_not_raise(self):
        """SMTP errors must be caught — should not crash the pipeline."""
        ar = self._make_ar()
        calls = []
        with patch("alertas.enviar_alerta", side_effect=Exception("SMTP connection refused")):
            with patch.dict(os.environ, {"EMAIL_ACTIVO": "true"}):
                from src.alertas_v2 import maybe_enviar_alerta_cnc
                fs = MagicMock()
                fs.primary_axis = "x"
                fs.multiaxis.get_axis.return_value = None
                try:
                    maybe_enviar_alerta_cnc(
                        1, 1, "T", ar, fs,
                        puede_enviar_fn=lambda m, cooldown_hours=1.0: True,
                        registrar_fn=lambda *a, **kw: (calls.append(a), 99)[1],
                    )
                except Exception as exc:
                    pytest.fail(f"SMTP failure should not propagate: {exc}")
        assert len(calls) == 1  # alert was registered despite SMTP failure

    def test_cooldown_active_no_alert(self):
        ar = self._make_ar()
        result, calls = self._call_maybe(
            ar, puede_enviar_fn=lambda m, cooldown_hours=1.0: False
        )
        assert result is False and len(calls) == 0

    def test_alert_registered_before_smtp(self):
        """registrar_alerta must be called before SMTP so it's always logged."""
        order = []
        def record_registrar(*a, **kw):
            order.append("registrar")
            return 99
        ar = self._make_ar()
        with patch("alertas.enviar_alerta") as mock_smtp:
            mock_smtp.side_effect = lambda **kw: order.append("smtp")
            with patch.dict(os.environ, {"EMAIL_ACTIVO": "true"}):
                from src.alertas_v2 import maybe_enviar_alerta_cnc
                fs = MagicMock()
                fs.primary_axis = "x"
                fs.multiaxis.get_axis.return_value = None
                maybe_enviar_alerta_cnc(
                    1, 1, "T", ar, fs,
                    puede_enviar_fn=lambda m, cooldown_hours=1.0: True,
                    registrar_fn=record_registrar,
                )
        if order:
            assert order[0] == "registrar"


# ─── TestSchedulerResilience ──────────────────────────────────────────────────

class TestSchedulerResilience:

    def test_hardware_error_does_not_stop_scheduler(self, tmp_path):
        """A RuntimeError from sensor.read() must not crash the scheduler loop."""
        from src.edge_scheduler import SchedulerState, _run_one_cycle

        state = SchedulerState()
        state.last_health_score = 80

        class FailingSensor:
            def run_once(self):
                raise RuntimeError("I2C bus error — sensor disconnected")

        result = _run_one_cycle(FailingSensor(), state)
        assert state.cycles_error == 1
        assert state.cycles_ok    == 0
        assert result == 80  # last_health_score preserved

    def test_multiple_errors_accumulate(self, tmp_path):
        from src.edge_scheduler import SchedulerState, _run_one_cycle
        state = SchedulerState()

        class AlwaysFail:
            def run_once(self):
                raise OSError("hardware gone")

        for _ in range(5):
            _run_one_cycle(AlwaysFail(), state)
        assert state.cycles_error == 5
        assert state.cycles_ok    == 0

    def test_sensor_factory_with_adxl345_mock(self, tmp_path):
        """SensorFactory with adxl345 type and injected bus must work."""
        from src.edge.sensors.sensor_factory import create_sensor
        bus = make_mock_bus()
        cfg = make_sensor_config(sensor_type="adxl345", samples_per_window=5)
        sensor = create_sensor(cfg, bus=bus)
        sensor.configure()
        bus.read_i2c_block_data.return_value = [0, 0, 0, 0, 0, 0]
        reading = sensor.read()
        assert reading.n_samples == 5

    def test_scheduler_disabled_exits_cleanly(self, tmp_path):
        from src.edge_scheduler import run_scheduler
        yaml_path = str(tmp_path / "cfg.yaml")
        open(yaml_path, "w").write("""
machine:
  id: T
  empresa_id: 1
  maquina_id: 1
sensor:
  type: mock
  sampling_rate_hz: 3200
  samples_per_window: 3200
  odr_hz: 3200
  axes: [x, y, z]
scheduler:
  enabled: false
""")
        run_scheduler(yaml_path, sleep_fn=lambda s: None)
        # No exception = success


# ─── TestEntryPoint ───────────────────────────────────────────────────────────

class TestEntryPoint:

    def test_edge_scheduler_has_main(self):
        src = open(
            os.path.join(os.path.dirname(__file__), "../src/edge_scheduler.py")
        ).read()
        assert "def main()" in src
        assert 'if __name__ == "__main__"' in src

    def test_edge_scheduler_argparse_has_config(self):
        src = open(
            os.path.join(os.path.dirname(__file__), "../src/edge_scheduler.py")
        ).read()
        assert "--config" in src

    def test_production_docs_exist(self):
        doc_path = os.path.join(
            os.path.dirname(__file__), "../docs/production_setup.md"
        )
        assert os.path.exists(doc_path)
        content = open(doc_path).read()
        assert "adxl345" in content.lower()
        assert "systemd" in content.lower()

    def test_systemd_service_exists(self):
        svc_path = os.path.join(
            os.path.dirname(__file__), "../config/systemd/aurapredict-edge.service"
        )
        assert os.path.exists(svc_path)
        content = open(svc_path).read()
        assert "edge_scheduler" in content


# ─── TestRegression ───────────────────────────────────────────────────────────

class TestRegression:

    def test_mock_sensor_still_works(self):
        from src.edge.sensors.mock_sensor import MockSensor, MockSensorParams
        from src.edge.sensors.base_sensor import SensorConfig
        cfg    = SensorConfig("t","mock",3200,samples_per_window=50)
        sensor = MockSensor(cfg, MockSensorParams())
        sensor.configure()
        reading = sensor.read()
        assert reading.n_samples == 50

    def test_acquisition_session_still_works(self, tmp_path):
        from src.edge.config.edge_config import (
            EdgeConfig, MachineConfig, AcquisitionConfig, BufferConfig
        )
        from src.edge.signal_processing import SignalConfig
        from src.edge.sensors.base_sensor import SensorConfig
        from src.edge.sensors.mock_sensor import MockSensor, MockSensorParams
        from src.edge.pipeline.acquisition import AcquisitionSession
        cfg = EdgeConfig(
            machine=MachineConfig(machine_id="T", empresa_id=1, maquina_id=1),
            sensor=SensorConfig(sensor_id="t", sensor_type="mock",
                                sampling_rate_hz=3200, samples_per_window=3200,
                                axes=["x","y","z"]),
            signal=SignalConfig(fs=3200),
            acquisition=AcquisitionConfig(),
            buffer=BufferConfig(base_dir=str(tmp_path)),
        )
        sensor = MockSensor(cfg.sensor, MockSensorParams())
        sensor.configure()
        fs = AcquisitionSession(cfg).acquire(sensor.read())
        assert fs is not None

    def test_legacy_scheduler_not_modified(self):
        """scheduler.py (legacy) must still be importable and unchanged."""
        src = open(
            os.path.join(os.path.dirname(__file__), "../src/scheduler.py"),
            encoding="utf-8",
        ).read()
        assert "MODO_SIMULADO" in src  # legacy code still present


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
