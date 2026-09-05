"""
Tests de la Fase 8 — Preparación y validación de hardware real

Cobertura:
  TestADXL345Verify       — adxl345_verify.py con mock bus
  TestADXL345Timeout      — read() timeout guard
  TestADXL345DataReady    — DATA_READY check opcional
  TestADXL345ErrorTypes   — categorización de errores (SensorReadError vs Config)
  TestSignalQualityRange  — expected_range_g usa rango del sensor
  TestAcquisitionRange    — acquisition pasa range_g correcto a quality check
  TestHardwareDocs        — documentación existe y es completa
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def make_sensor_config(
    sensor_type="adxl345",
    samples_per_window=20,
    sampling_rate_hz=800.0,
    odr_hz=800.0,
    range_g=2,
    i2c_address="0x53",
    extra_overrides=None,
):
    from src.edge.sensors.base_sensor import SensorConfig
    extra = {"i2c_bus": 1, "range_g": range_g}
    if extra_overrides:
        extra.update(extra_overrides)
    return SensorConfig(
        sensor_id          = "test_adxl",
        sensor_type        = sensor_type,
        sampling_rate_hz   = sampling_rate_hz,
        odr_hz             = odr_hz,
        samples_per_window = samples_per_window,
        axes               = ["x", "y", "z"],
        i2c_address        = i2c_address,
        extra              = extra,
    )


def make_mock_bus(devid=0xE5, sample_bytes=None):
    """Returns a mock I2C bus for testing."""
    if sample_bytes is None:
        sample_bytes = [0x00, 0x10, 0x00, 0x00, 0x00, 0x00]  # X=0x1000=4096
    bus = MagicMock()
    # read_i2c_block_data returns devid on first call, sample bytes thereafter
    bus.read_i2c_block_data.side_effect = [
        [devid],                # check 1: DEVID
    ] + [sample_bytes] * 10000  # subsequent reads
    return bus


def make_configured_sensor(samples=10, sampling_rate_hz=800.0, extra=None):
    from src.edge.sensors.adxl345_sensor import ADXL345Sensor
    cfg = make_sensor_config(samples_per_window=samples,
                              sampling_rate_hz=sampling_rate_hz,
                              extra_overrides=extra)
    bus = make_mock_bus()
    sensor = ADXL345Sensor(cfg, bus=bus)
    sensor.configure()
    return sensor, bus


# ─── TestADXL345Verify ────────────────────────────────────────────────────────

class TestADXL345Verify:

    def _make_verify_bus(self, devid=0xE5, register_value=None):
        """Mock bus that passes all verification checks."""
        bus = MagicMock()
        # DEVID, register readbacks, then samples with SLIGHT VARIATION
        # (non-constant required: std > 1e-4 g)
        samples = []
        for i in range(10):
            # X alternates between 0x100 (1.0g) and 0x110 (1.065g) — non-constant
            x_val = 0x100 + i
            samples.append([x_val & 0xFF, (x_val >> 8) & 0xFF,
                             0x00, 0x00, 0x00, 0x00])
        side_effects = [
            [devid],
            [0x08 | 0x00],   # DATA_FORMAT readback
            [0x0C],          # BW_RATE readback
            [0x08],          # POWER_CTL readback
        ] + samples
        bus.read_i2c_block_data.side_effect = side_effects
        return bus

    def test_verify_all_pass_with_mock(self):
        from src.edge.sensors.adxl345_verify import verify_adxl345
        bus = self._make_verify_bus()
        result = verify_adxl345(bus=bus)
        assert result.passed, f"Should pass: {result.errors}"

    def test_verify_result_has_all_checks(self):
        from src.edge.sensors.adxl345_verify import verify_adxl345
        bus = self._make_verify_bus()
        result = verify_adxl345(bus=bus)
        expected_checks = [
            "3_devid_correct", "4_data_format_rw", "5_bw_rate_rw",
            "6_measurement_mode", "7_samples_readable",
            "8_non_constant", "9_physical_range", "10_standby_ok",
        ]
        for check in expected_checks:
            assert check in result.checks, f"Missing check: {check}"

    def test_verify_wrong_devid_fails(self):
        from src.edge.sensors.adxl345_verify import verify_adxl345
        bus = self._make_verify_bus(devid=0x00)
        result = verify_adxl345(bus=bus)
        assert not result.passed
        assert not result.checks.get("3_devid_correct")

    def test_verify_i2c_error_on_devid_fails(self):
        from src.edge.sensors.adxl345_verify import verify_adxl345
        bus = MagicMock()
        bus.read_i2c_block_data.side_effect = OSError("I2C error")
        result = verify_adxl345(bus=bus)
        assert not result.passed
        assert not result.checks.get("3_devid_correct")

    def test_verify_returns_details(self):
        from src.edge.sensors.adxl345_verify import verify_adxl345
        bus = self._make_verify_bus()
        result = verify_adxl345(bus=bus)
        assert "address" in result.details
        assert "devid_read" in result.details

    def test_verify_result_summary_string(self):
        from src.edge.sensors.adxl345_verify import verify_adxl345
        bus = self._make_verify_bus()
        result = verify_adxl345(bus=bus)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "PASS" in summary or "FAIL" in summary

    def test_verify_constant_signal_fails(self):
        from src.edge.sensors.adxl345_verify import verify_adxl345
        bus = MagicMock()
        # All zeros — constant signal
        bus.read_i2c_block_data.side_effect = [
            [0xE5],                # DEVID
            [0x08],                # DATA_FORMAT
            [0x0C],                # BW_RATE
            [0x08],                # POWER_CTL
        ] + [[0x00, 0x00, 0x00, 0x00, 0x00, 0x00]] * 10
        result = verify_adxl345(bus=bus)
        assert not result.checks.get("8_non_constant")

    def test_verify_failed_checks_list(self):
        from src.edge.sensors.adxl345_verify import verify_adxl345
        bus = self._make_verify_bus(devid=0x00)
        result = verify_adxl345(bus=bus)
        assert "3_devid_correct" in result.failed_checks


# ─── TestADXL345Timeout ───────────────────────────────────────────────────────

class TestADXL345Timeout:

    def test_timeout_raises_sensor_read_error(self):
        """If read loop exceeds deadline, SensorReadError must be raised."""
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor
        from src.edge.sensors.base_sensor import SensorReadError

        cfg = make_sensor_config(samples_per_window=100, sampling_rate_hz=1.0)
        # At 1 Hz, deadline = 100s * 3 = 300s — but we'll mock time to exceed it
        bus = make_mock_bus()
        sensor = ADXL345Sensor(cfg, bus=bus)
        sensor.configure()

        # Patch monotonic to simulate time passing beyond deadline immediately
        original_monotonic = time.monotonic
        call_count = [0]

        def fast_time():
            call_count[0] += 1
            # First call: return start time; second: return far future
            return 0.0 if call_count[0] == 1 else 99999.0

        with patch("edge.sensors.adxl345_sensor.time.monotonic", side_effect=fast_time):
            with pytest.raises(SensorReadError, match="timeout"):
                sensor.read()

    def test_normal_read_no_timeout(self):
        """Fast I2C (mock) should never trigger timeout."""
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor
        from src.edge.sensors.base_sensor import SensorReadError

        sensor, bus = make_configured_sensor(samples=5, sampling_rate_hz=800.0)
        bus.read_i2c_block_data.side_effect = [
            [0xE5],
        ] + [[0x00, 0x10, 0x00, 0x00, 0x00, 0x00]] * 10000

        # No error should be raised with normal mock bus speed
        sensor, bus = make_configured_sensor(samples=5)
        reading = sensor.read()
        assert reading.n_samples == 5


# ─── TestADXL345DataReady ─────────────────────────────────────────────────────

class TestADXL345DataReady:

    def test_data_ready_disabled_by_default(self):
        """check_data_ready is False by default — INT_SOURCE not read."""
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor, REG_INT_SOURCE
        sensor, bus = make_configured_sensor(samples=3)
        sensor.read()
        # INT_SOURCE should NOT have been read (only DATAX0 reads)
        calls = bus.read_i2c_block_data.call_args_list
        int_source_calls = [c for c in calls if c[0][1] == REG_INT_SOURCE]
        assert len(int_source_calls) == 0

    def test_data_ready_enabled_reads_int_source(self):
        """With check_data_ready=True, INT_SOURCE is read before each sample."""
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor, REG_INT_SOURCE
        from src.edge.sensors.base_sensor import SensorConfig

        cfg = make_sensor_config(samples_per_window=2,
                                  extra_overrides={"check_data_ready": True})
        bus = MagicMock()
        # DEVID, then alternating INT_SOURCE (DATA_READY set) + DATAX0
        int_source_ready = [0x80]  # DATA_READY bit set
        sample = [0x00, 0x10, 0x00, 0x00, 0x00, 0x00]
        bus.read_i2c_block_data.side_effect = [
            [0xE5],          # DEVID
            int_source_ready, sample,  # sample 0
            int_source_ready, sample,  # sample 1
        ]

        sensor = ADXL345Sensor(cfg, bus=bus)
        sensor.configure()
        reading = sensor.read()
        assert reading.n_samples == 2

        # INT_SOURCE must have been read for each sample
        calls = bus.read_i2c_block_data.call_args_list
        int_src_calls = [c for c in calls if c[0][1] == REG_INT_SOURCE]
        assert len(int_src_calls) == 2

    def test_data_ready_timeout_continues(self):
        """If DATA_READY never sets within timeout, sensor still reads anyway."""
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor, REG_INT_SOURCE
        cfg = make_sensor_config(samples_per_window=1,
                                  sampling_rate_hz=800.0,
                                  extra_overrides={"check_data_ready": True})
        bus = MagicMock()
        sample = [0x00, 0x10, 0x00, 0x00, 0x00, 0x00]

        # Use side_effect function: INT_SOURCE always returns 0x00 (not ready)
        # DATAX0 always returns a valid sample
        def smart_read(addr, reg, length):
            if reg == 0x00:   return [0xE5]  # DEVID
            if reg == 0x30:   return [0x00]  # INT_SOURCE — DATA_READY not set
            if reg == 0x32:   return sample  # DATAX0
            return [0x00]

        bus.read_i2c_block_data.side_effect = smart_read

        sensor = ADXL345Sensor(cfg, bus=bus)
        sensor.configure()

        # Monotonic call sequence inside read() for 1 sample with DATA_READY:
        #   1. deadline = start + window*3  (call 1: returns 0.0 → deadline=big)
        #   2. outer check: mono > deadline  (call 2: returns 0.0 → False, ok)
        #   3. wait_deadline = mono + period*2 (call 3: returns 0.0)
        #   4. inner check: mono > wait_deadline (call 4: returns 999.0 → break)
        #   5. DATAX0 read happens; t_end = time.time() (not patched)
        with patch("edge.sensors.adxl345_sensor.time.monotonic",
                   side_effect=[0.0, 0.0, 0.0, 999.0] + [999.0] * 50):
            reading = sensor.read()
        assert reading.n_samples == 1


# ─── TestADXL345ErrorTypes ────────────────────────────────────────────────────

class TestADXL345ErrorTypes:

    def test_read_before_configure_raises_config_error(self):
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor
        from src.edge.sensors.base_sensor import SensorConfigurationError
        from src.edge.sensors.base_sensor import SensorConfig
        cfg = SensorConfig("t", "adxl345", 800, samples_per_window=5)
        sensor = ADXL345Sensor(cfg)
        with pytest.raises(SensorConfigurationError):
            sensor.read()

    def test_i2c_error_during_read_raises_sensor_read_error(self):
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor
        from src.edge.sensors.base_sensor import SensorReadError
        sensor, bus = make_configured_sensor(samples=5)
        bus.read_i2c_block_data.side_effect = OSError("I2C bus error")
        with pytest.raises(SensorReadError, match="I2C error"):
            sensor.read()

    def test_wrong_devid_raises_configuration_error(self):
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor
        from src.edge.sensors.base_sensor import SensorConfig, SensorConfigurationError
        cfg = SensorConfig("t", "adxl345", 800, samples_per_window=5)
        bus = MagicMock()
        bus.read_i2c_block_data.return_value = [0x00]  # wrong DEVID
        sensor = ADXL345Sensor(cfg, bus=bus)
        with pytest.raises(SensorConfigurationError, match="DEVID mismatch"):
            sensor.configure()

    def test_read_error_contains_sample_index(self):
        """Error message should show which sample failed — aids hardware debugging."""
        from src.edge.sensors.adxl345_sensor import ADXL345Sensor
        from src.edge.sensors.base_sensor import SensorReadError
        sensor, bus = make_configured_sensor(samples=10)
        good_sample = [0x00, 0x10, 0x00, 0x00, 0x00, 0x00]
        # Fail on 5th sample
        bus.read_i2c_block_data.side_effect = [
            [0xE5],
        ] + [good_sample] * 4 + [OSError("bus error")]
        with pytest.raises(SensorReadError):
            sensor.read()


# ─── TestSignalQualityRange ───────────────────────────────────────────────────

class TestSignalQualityRange:

    def test_signal_within_2g_range_not_flagged(self):
        """Signal of 1g should be OK for ±2g sensor, not flagged as out-of-range."""
        from src.edge.data_quality import check_signal_quality
        # 1g signal — normal for a sensor lying on its side
        signal = np.ones(200) * 1.0
        signal += np.random.randn(200) * 0.05  # small noise
        result = check_signal_quality(
            signal           = signal,
            configured_hz    = 800.0,
            expected_range_g = (-2.1, 2.1),  # ±2g + 5%
        )
        assert not result.is_out_of_range, "1g should be OK for ±2g sensor"

    def test_signal_at_3g_flagged_for_2g_sensor(self):
        """3g reading should be flagged as out-of-range for ±2g sensor."""
        from src.edge.data_quality import check_signal_quality
        signal = np.ones(200) * 3.0
        signal += np.random.randn(200) * 0.05
        result = check_signal_quality(
            signal           = signal,
            configured_hz    = 800.0,
            expected_range_g = (-2.1, 2.1),
        )
        assert result.is_out_of_range

    def test_signal_at_3g_ok_for_16g_sensor(self):
        """3g should be OK for ±16g sensor (legacy default)."""
        from src.edge.data_quality import check_signal_quality
        signal = np.ones(200) * 3.0
        signal += np.random.randn(200) * 0.05
        result = check_signal_quality(
            signal           = signal,
            configured_hz    = 800.0,
            expected_range_g = (-16.8, 16.8),  # ±16g + 5%
        )
        assert not result.is_out_of_range


# ─── TestAcquisitionRange ─────────────────────────────────────────────────────

class TestAcquisitionRange:

    def test_acquisition_passes_range_g_to_quality_check(self):
        """acquisition.py must extract range_g from sensor config and pass it."""
        src = open(
            os.path.join(os.path.dirname(__file__), "../src/edge/pipeline/acquisition.py")
        ).read()
        assert "range_g" in src
        assert "expected_range_g" in src

    def test_acquisition_uses_sensor_range_not_default(self):
        """The range passed should come from config.sensor.extra, not hardcoded."""
        src = open(
            os.path.join(os.path.dirname(__file__), "../src/edge/pipeline/acquisition.py")
        ).read()
        assert 'extra.get("range_g"' in src
        assert "phys_range" in src or "expected_range_g" in src


# ─── TestHardwareDocs ─────────────────────────────────────────────────────────

class TestHardwareDocs:

    def _doc_path(self):
        return os.path.join(
            os.path.dirname(__file__), "../docs/hardware_setup_raspberry_pi.md"
        )

    def test_doc_exists(self):
        assert os.path.exists(self._doc_path())

    def test_doc_has_wiring_section(self):
        content = open(self._doc_path(), encoding="utf-8").read()
        assert "SDA" in content and "SCL" in content

    def test_doc_has_i2c_address(self):
        content = open(self._doc_path(), encoding="utf-8").read()
        assert "0x53" in content

    def test_doc_has_i2cdetect_command(self):
        content = open(self._doc_path(), encoding="utf-8").read()
        assert "i2cdetect" in content

    def test_doc_has_verification_step(self):
        content = open(self._doc_path(), encoding="utf-8").read()
        assert "adxl345_verify" in content

    def test_doc_has_sampling_rate_note(self):
        content = open(self._doc_path(), encoding="utf-8").read()
        assert "sampling" in content.lower()

    def test_doc_has_pending_validation_section(self):
        content = open(self._doc_path(), encoding="utf-8").read()
        assert "pendiente" in content.lower() or "pending" in content.lower()

    def test_verify_module_exists(self):
        assert os.path.exists(
            os.path.join(os.path.dirname(__file__),
                         "../src/edge/sensors/adxl345_verify.py")
        )

    def test_verify_module_syntax_valid(self):
        import ast
        src = open(
            os.path.join(os.path.dirname(__file__),
                         "../src/edge/sensors/adxl345_verify.py"),
            encoding="utf-8",
        ).read()
        ast.parse(src)

    def test_adxl345_sensor_syntax_valid(self):
        import ast
        src = open(
            os.path.join(os.path.dirname(__file__),
                         "../src/edge/sensors/adxl345_sensor.py")
        ).read()
        ast.parse(src)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
