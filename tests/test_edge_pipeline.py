"""
Tests del pipeline Edge: MockSensor → DataQuality → SignalProcessing → FeatureExtraction

Cubre los 11 escenarios requeridos:
  1. Señal normal
  2. Señal con ruido
  3. Señal con frecuencia dominante conocida
  4. Señal con aumento progresivo de vibración
  5. Señal plana (sensor desconectado)
  6. Señal saturada
  7. NaN / Inf
  8. Pérdida de muestras
  9. Sampling rate mismatch
 10. Los tres ejes X/Y/Z independientes
 11. Cada modo del MockSensor

Todos los tests son deterministas (seeds fijos).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.edge.sensors.base_sensor import SensorConfig, SensorReading
from src.edge.sensors.mock_sensor import MockSensor, MockSensorParams, SignalMode
from src.edge.data_quality import (
    DataQualityResult,
    SamplingRateResult,
    check_signal_quality,
    detect_sampling_rate,
)
from src.edge.feature_extraction import (
    TimeFeatures,
    FreqFeatures,
    VibrationFeatures,
    OperatingContext,
    OrderAnalysisPrep,
    MultiAxisReading,
    extract_vibration_features,
    extract_multiaxis_features,
)
from src.edge.signal_processing import SignalConfig


# ─── CONSTANTES Y HELPERS ─────────────────────────────────────────────────────

FS      = 3200.0
N       = 3200     # 1 segundo de datos
AXES    = ["x", "y", "z"]


def make_config(sensor_id: str = "mock_test") -> SensorConfig:
    return SensorConfig(
        sensor_id          = sensor_id,
        sensor_type        = "mock",
        sampling_rate_hz   = FS,
        odr_hz             = FS,
        samples_per_window = N,
        axes               = AXES,
    )


def make_sensor(mode: SignalMode = SignalMode.NORMAL,
                amplitude: float = 0.07,
                freq: float = 50.0,
                noise: float = 0.005,
                seed: int = 42) -> MockSensor:
    params = MockSensorParams(
        mode               = mode,
        amplitude_g        = amplitude,
        dominant_freq_hz   = freq,
        noise_amplitude_g  = noise,
        seed               = seed,
        axis_amplitude_factor = {"x": 1.0, "y": 0.6, "z": 0.3},
    )
    return MockSensor(make_config(), params)


def gen_sine(freq: float = 50.0, amp: float = 0.07,
             fs: float = FS, n: int = N, seed: int = 0) -> np.ndarray:
    t = np.linspace(0, n / fs, n, endpoint=False)
    rng = np.random.default_rng(seed)
    return amp * np.sin(2 * np.pi * freq * t) + rng.normal(0, 0.001, n)


# ─── TESTS: DATA QUALITY ──────────────────────────────────────────────────────

class TestDataQuality:
    """Tests del módulo data_quality.py."""

    def test_1_normal_signal_valid(self):
        """Señal normal debe pasar con calidad alta."""
        signal = gen_sine()
        result = check_signal_quality(signal, FS)
        assert result.is_valid is True
        assert result.is_sensor_error is False
        assert result.quality_score >= 0.8
        assert result.status == "OK"

    def test_5_flat_signal_sensor_error(self):
        """Señal plana → SENSOR_ERROR, nunca anomalía de máquina."""
        flat = np.zeros(N)
        result = check_signal_quality(flat, FS)
        assert result.is_valid is False
        assert result.is_sensor_error is True
        assert result.is_flat is True
        assert result.status == "SENSOR_ERROR"
        # Verificar que el mensaje deja claro que NO es fallo de máquina
        assert any("SENSOR_ERROR" in e or "sensor" in e.lower()
                   for e in result.errors)

    def test_7_nan_signal_invalid(self):
        """NaN en la señal → SENSOR_ERROR."""
        signal = gen_sine()
        signal[500] = np.nan
        result = check_signal_quality(signal, FS)
        assert result.is_valid is False
        assert result.is_sensor_error is True
        assert result.has_nan_inf is True

    def test_7_inf_signal_invalid(self):
        """Inf en la señal → SENSOR_ERROR."""
        signal = gen_sine()
        signal[200] = np.inf
        result = check_signal_quality(signal, FS)
        assert result.is_valid is False
        assert result.is_sensor_error is True
        assert result.has_nan_inf is True

    def test_6_saturated_signal_warns(self):
        """Señal saturada → calidad baja, advertencia."""
        t = np.linspace(0, 1, N, endpoint=False)
        raw = 2.0 * np.sin(2 * np.pi * 50 * t)  # amplitud > rango
        saturated = np.clip(raw, -0.5, 0.5)
        result = check_signal_quality(saturated, FS)
        assert result.is_saturated is True
        assert result.quality_score < 0.7
        assert len(result.warnings) > 0

    def test_9_sampling_rate_mismatch_detected(self):
        """Señal de 1600 Hz presentada como 3200 Hz → mismatch detectado."""
        # Generar señal a la mitad de la tasa esperada
        n_actual = N // 2
        t_actual = np.linspace(0, 0.5, n_actual, endpoint=False)
        signal_short = 0.07 * np.sin(2 * np.pi * 50 * t_actual)

        # Timestamps que indican 1600 Hz real pero configurado 3200 Hz
        ts_actual = np.linspace(0, 1.0, n_actual, endpoint=False)  # 1 segundo real

        result = check_signal_quality(
            signal_short, FS, timestamps=ts_actual, duration_s=1.0
        )
        assert result.sampling_rate is not None
        assert result.sampling_rate.mismatch_detected is True
        assert result.sampling_rate.actual_hz is not None
        assert abs(result.sampling_rate.actual_hz - 1600) < 100

    def test_8_sample_loss_detected(self):
        """Pérdida de muestras detectada cuando se reciben menos de las esperadas."""
        # Configurado: 3200 Hz × 1 s = 3200 muestras
        # Recibidas: solo 2500 (21% de pérdida)
        n_received = 2500
        signal     = gen_sine(n=n_received)

        # Timestamps: 2500 muestras en 1 segundo (loss visible)
        ts = np.linspace(0, 1.0, n_received, endpoint=False)
        result = check_signal_quality(signal, FS, timestamps=ts, duration_s=1.0)
        assert result.sampling_rate is not None
        sr = result.sampling_rate
        assert sr.sample_loss_fraction is not None
        assert sr.sample_loss_fraction > 0.15   # > 15% de pérdida

    def test_timestamp_anomaly_nonmonotonic(self):
        """Timestamps no monótonos → anomalía detectada."""
        signal = gen_sine()
        ts = np.linspace(0, 1, N, endpoint=False)
        ts[500] = ts[400]  # timestamp duplicado
        result = check_signal_quality(signal, FS, timestamps=ts)
        assert result.has_timestamp_anomaly is True

    def test_short_signal_rejected(self):
        """Señal de menos de 32 muestras → inválida."""
        tiny = np.ones(10)
        result = check_signal_quality(tiny, FS)
        assert result.is_valid is False
        assert result.is_sensor_error is True

    def test_out_of_range_warns(self):
        """Amplitud fuera del rango físico esperado → advertencia."""
        t = np.linspace(0, 1, N, endpoint=False)
        signal = 20.0 * np.sin(2 * np.pi * 50 * t)  # > 16g límite por defecto
        result = check_signal_quality(signal, FS, expected_range_g=(-16.0, 16.0))
        assert result.is_out_of_range is True
        assert len(result.warnings) > 0


class TestSamplingRateDetection:
    """Tests de detect_sampling_rate()."""

    def test_exact_rate_from_timestamps(self):
        """Timestamps perfectos deben detectar la tasa configurada."""
        ts = np.linspace(0, 1.0, N, endpoint=False)
        result = detect_sampling_rate(ts, N, configured_hz=FS, duration_s=1.0)
        assert result.actual_hz is not None
        assert abs(result.actual_hz - FS) < 10.0  # tolerancia 10 Hz

    def test_mismatch_flagged(self):
        """Tasa real distinta de configurada → mismatch_detected=True."""
        # 1600 muestras en 1 segundo = 1600 Hz, config dice 3200
        ts = np.linspace(0, 1.0, 1600, endpoint=False)
        result = detect_sampling_rate(ts, 1600, configured_hz=FS, duration_s=1.0)
        assert result.mismatch_detected is True
        assert result.warning is not None

    def test_no_timestamps_uses_duration(self):
        """Sin timestamps pero con duración → estimación desde sample count."""
        result = detect_sampling_rate(
            None, N, configured_hz=FS, duration_s=1.0
        )
        assert result.actual_hz is not None
        assert abs(result.actual_hz - FS) < 1.0

    def test_effective_hz_falls_back_correctly(self):
        """effective_hz usa actual si disponible, configured si no."""
        r1 = detect_sampling_rate(None, N, configured_hz=FS)
        assert r1.effective_hz == FS  # sin timestamps, usa configured

        ts = np.linspace(0, 0.5, N // 2, endpoint=False)
        r2 = detect_sampling_rate(ts, N // 2, configured_hz=FS, duration_s=0.5)
        assert r2.effective_hz == r2.actual_hz


# ─── TESTS: FEATURE EXTRACTION ────────────────────────────────────────────────

class TestFeatureExtraction:
    """Tests del módulo feature_extraction.py."""

    def test_1_normal_signal_features_finite(self):
        """Señal normal → todas las features son números finitos."""
        signal = gen_sine()
        vf = extract_vibration_features(signal, "x", FS)
        assert isinstance(vf, VibrationFeatures)
        assert np.isfinite(vf.time.rms)
        assert np.isfinite(vf.time.kurtosis)
        assert np.isfinite(vf.time.crest_factor)
        assert np.isfinite(vf.freq.dominant_freq)
        assert np.isfinite(vf.freq.spectral_energy)
        assert vf.axis == "x"

    def test_4_progressive_vibration_rms_increases(self):
        """Vibración progresiva → RMS aumenta con la amplitud."""
        rms_values = []
        for amp in [0.05, 0.15, 0.35, 0.70]:
            signal = gen_sine(amp=amp)
            vf = extract_vibration_features(signal, "x", FS)
            rms_values.append(vf.time.rms)
        # RMS debe ser estrictamente creciente
        for i in range(len(rms_values) - 1):
            assert rms_values[i] < rms_values[i + 1], \
                f"RMS no aumentó: {rms_values}"

    def test_3_dominant_frequency_detected(self):
        """Señal con frecuencia dominante → FFT la detecta correctamente."""
        target_hz = 150.0
        t = np.linspace(0, 1, N, endpoint=False)
        signal = (0.20 * np.sin(2 * np.pi * target_hz * t) +
                  0.02 * np.sin(2 * np.pi * 50.0 * t))
        vf = extract_vibration_features(signal, "x", FS)
        assert abs(vf.freq.dominant_freq - target_hz) < 3.0, \
            f"Esperada {target_hz} Hz, detectada {vf.freq.dominant_freq:.1f} Hz"

    def test_10_axes_independent(self):
        """Los ejes X/Y/Z deben producir features independientes y diferentes."""
        signals = {
            "x": gen_sine(amp=0.20, seed=1),
            "y": gen_sine(amp=0.12, seed=2),
            "z": gen_sine(amp=0.05, seed=3),
        }
        reading = extract_multiaxis_features(
            signals              = signals,
            sensor_id            = "test",
            sampling_rate_configured = FS,
        )
        assert reading.x is not None
        assert reading.y is not None
        assert reading.z is not None
        # RMS debe ser diferente (distintas amplitudes)
        assert abs(reading.x.time.rms - reading.y.time.rms) > 0.01
        assert abs(reading.y.time.rms - reading.z.time.rms) > 0.01
        # Los ejes no deben estar mezclados
        assert reading.x.axis == "x"
        assert reading.y.axis == "y"
        assert reading.z.axis == "z"

    def test_10_axes_not_combined(self):
        """Sin include_total=True, el eje 'total' debe ser None."""
        signals = {a: gen_sine(seed=i) for i, a in enumerate(("x", "y", "z"))}
        reading = extract_multiaxis_features(
            signals, "test", FS, include_total=False
        )
        assert reading.total is None

    def test_10_total_axis_explicit(self):
        """Con include_total=True, total debe ser computado explícitamente y separado de los ejes."""
        signals = {a: gen_sine(amp=0.1, seed=i) for i, a in enumerate(("x", "y", "z"))}
        reading = extract_multiaxis_features(
            signals, "test", FS, include_total=True
        )
        assert reading.total is not None
        assert reading.total.axis == "total"
        assert reading.total.time.rms > 0
        assert np.isfinite(reading.total.time.rms)
        # Los ejes individuales se conservan separados — no se reemplazan por total
        assert reading.x is not None and reading.x.axis == "x"
        assert reading.y is not None and reading.y.axis == "y"
        assert reading.z is not None and reading.z.axis == "z"
        # total y eje X son distintos (distintas propiedades espectrales)
        assert reading.total.axis != reading.x.axis

    def test_api_dict_compatibility(self):
        """to_api_dict debe producir el formato del endpoint POST /predict/bearing."""
        signals = {"x": gen_sine(), "y": gen_sine(amp=0.06), "z": gen_sine(amp=0.03)}
        reading = extract_multiaxis_features(signals, "Torno_CNC_1", FS)
        d = reading.to_api_dict(axis="x")
        assert set(d.keys()) == {"maquina", "RMS", "Peak_to_Peak", "Kurtosis", "Skewness"}
        assert d["maquina"] == "Torno_CNC_1"
        assert all(isinstance(d[k], float) for k in ["RMS", "Peak_to_Peak", "Kurtosis", "Skewness"])

    def test_band_energies_present(self):
        """Las bandas de frecuencia deben estar presentes y ser positivas."""
        signal = gen_sine(amp=0.1)
        vf     = extract_vibration_features(signal, "x", FS)
        assert len(vf.freq.band_energies) > 0
        assert all(v >= 0 for v in vf.freq.band_energies.values())


class TestOperatingContext:
    """Tests de OperatingContext — separación rpm_nominal vs rpm_real."""

    def test_default_all_none(self):
        """Por defecto todo debe ser None."""
        ctx = OperatingContext()
        assert ctx.rpm_nominal is None
        assert ctx.rpm_real is None
        assert ctx.rpm_source is None
        assert ctx.temperatura is None
        assert ctx.carga is None

    def test_rpm_real_available_false_without_rpm(self):
        """Sin rpm_real, rpm_real_available debe ser False."""
        ctx = OperatingContext(rpm_nominal=3000.0)
        assert ctx.rpm_real_available is False

    def test_rpm_real_available_true_with_rpm(self):
        """Con rpm_real, rpm_real_available debe ser True."""
        ctx = OperatingContext(rpm_nominal=3000.0, rpm_real=2950.0, rpm_source="encoder")
        assert ctx.rpm_real_available is True

    def test_rpm_for_analysis_returns_none_without_real(self):
        """rpm_for_analysis no usa rpm_nominal como sustituto de rpm_real."""
        ctx = OperatingContext(rpm_nominal=3000.0)
        rpm, src = ctx.rpm_for_analysis()
        assert rpm is None
        assert src == "not_available"

    def test_rpm_for_analysis_returns_real_when_available(self):
        """rpm_for_analysis retorna rpm_real cuando está disponible."""
        ctx = OperatingContext(rpm_nominal=3000.0, rpm_real=2950.0, rpm_source="encoder")
        rpm, src = ctx.rpm_for_analysis()
        assert rpm == 2950.0
        assert "encoder" in src


class TestOrderAnalysisPrep:
    """Tests de OrderAnalysisPrep."""

    def test_not_available_without_rpm_real(self):
        """Sin rpm_real, order analysis no disponible aunque haya rpm_nominal."""
        ctx  = OperatingContext(rpm_nominal=3000.0)  # nominal no cuenta
        prep = OrderAnalysisPrep(context=ctx)
        assert prep.is_available is False

    def test_order_frequency_none_without_rpm(self):
        """order_frequency devuelve None cuando rpm_real no está disponible."""
        ctx  = OperatingContext(rpm_nominal=3000.0)
        prep = OrderAnalysisPrep(context=ctx)
        assert prep.order_frequency(1) is None
        assert prep.order_frequency(2) is None
        assert prep.order_frequency(3) is None

    def test_order_frequency_correct_with_rpm(self):
        """Con rpm_real=3000 → 1X=50Hz, 2X=100Hz, 3X=150Hz."""
        ctx  = OperatingContext(rpm_real=3000.0, rpm_source="encoder")
        prep = OrderAnalysisPrep(context=ctx)
        assert prep.is_available is True
        assert abs(prep.order_frequency(1) - 50.0)  < 0.01
        assert abs(prep.order_frequency(2) - 100.0) < 0.01
        assert abs(prep.order_frequency(3) - 150.0) < 0.01

    def test_order_frequencies_dict(self):
        """order_frequencies() devuelve dict con valores None sin rpm_real."""
        ctx  = OperatingContext()
        prep = OrderAnalysisPrep(context=ctx)
        freqs = prep.order_frequencies()
        assert all(v is None for v in freqs.values())

    def test_order_frequencies_dict_with_rpm(self):
        """order_frequencies() devuelve dict con valores correctos con rpm_real."""
        ctx  = OperatingContext(rpm_real=1800.0, rpm_source="opc_ua")
        prep = OrderAnalysisPrep(context=ctx, orders=[1, 2])
        freqs = prep.order_frequencies()
        assert abs(freqs["1X"] - 30.0) < 0.01   # 1800/60 = 30 Hz
        assert abs(freqs["2X"] - 60.0) < 0.01


# ─── TESTS: MOCK SENSOR ───────────────────────────────────────────────────────

class TestMockSensor:
    """Tests del MockSensor — todos los modos."""

    def test_configure_and_read(self):
        """El sensor debe configurarse y generar lecturas."""
        sensor = make_sensor()
        sensor.configure()
        reading = sensor.read()
        assert isinstance(reading, SensorReading)
        assert reading.n_samples == N
        assert set(AXES).issubset(reading.available_axes)
        sensor.close()

    def test_context_manager(self):
        """Debe funcionar como context manager."""
        with make_sensor() as sensor:
            reading = sensor.read()
        assert reading.n_samples == N

    def test_reproducible_same_seed(self):
        """Con el mismo seed, dos lecturas deben ser idénticas."""
        with make_sensor(seed=42) as s1:
            r1 = s1.read()
        with make_sensor(seed=42) as s2:
            r2 = s2.read()
        np.testing.assert_array_equal(r1.axes["x"], r2.axes["x"])
        np.testing.assert_array_equal(r1.axes["y"], r2.axes["y"])

    def test_different_seeds_different_signals(self):
        """Seeds distintos deben producir señales diferentes."""
        with make_sensor(seed=42) as s1:
            r1 = s1.read()
        with make_sensor(seed=99) as s2:
            r2 = s2.read()
        assert not np.allclose(r1.axes["x"], r2.axes["x"])

    def test_11_normal_mode(self):
        """NORMAL: RMS bajo, kurtosis baja, señal válida."""
        with make_sensor(mode=SignalMode.NORMAL) as s:
            reading = s.read()
        sig = reading.axes["x"]
        qr  = check_signal_quality(sig, FS)
        vf  = extract_vibration_features(sig, "x", FS)
        assert qr.is_valid
        assert not qr.is_sensor_error
        assert vf.time.rms < 0.15
        assert abs(vf.time.kurtosis) < 2.0   # señal senoidal ≈ -1.5 (Fisher)

    def test_11_imbalance_mode_higher_rms(self):
        """IMBALANCE: RMS mayor que NORMAL por mayor amplitud."""
        with make_sensor(mode=SignalMode.NORMAL) as s:
            r_normal = s.read()
        with make_sensor(mode=SignalMode.IMBALANCE) as s:
            r_imbalance = s.read()

        rms_n = float(np.sqrt(np.mean(r_normal.axes["x"] ** 2)))
        rms_i = float(np.sqrt(np.mean(r_imbalance.axes["x"] ** 2)))
        assert rms_i > rms_n, f"IMBALANCE ({rms_i:.4f}) debe ser > NORMAL ({rms_n:.4f})"

    def test_11_misalignment_mode_harmonics(self):
        """MISALIGNMENT: energía significativa en 2× la frecuencia fundamental."""
        with make_sensor(mode=SignalMode.MISALIGNMENT, freq=50.0) as s:
            reading = s.read()
        sig = reading.axes["x"]
        vf  = extract_vibration_features(sig, "x", FS)
        # La frecuencia dominante puede ser 50 Hz o 100 Hz (2× = misalignment)
        assert vf.freq.dominant_freq in range(40, 120), \
            f"Frecuencia dominante inesperada: {vf.freq.dominant_freq:.1f} Hz"

    def test_11_looseness_mode_high_peak_to_peak(self):
        """LOOSENESS: alta amplitud pico a pico por impactos aleatorios."""
        with make_sensor(mode=SignalMode.NORMAL) as s:
            r_normal = s.read()
        with make_sensor(mode=SignalMode.LOOSENESS) as s:
            r_loose = s.read()
        p2p_n = float(np.ptp(r_normal.axes["x"]))
        p2p_l = float(np.ptp(r_loose.axes["x"]))
        assert p2p_l > p2p_n, f"LOOSENESS P2P ({p2p_l:.4f}) debe ser > NORMAL ({p2p_n:.4f})"

    def test_11_bearing_degradation_high_kurtosis(self):
        """BEARING_DEGRADATION: kurtosis elevada por patrón impulsivo."""
        from scipy.stats import kurtosis as sp_kurtosis
        with make_sensor(mode=SignalMode.NORMAL) as s:
            r_normal = s.read()
        with make_sensor(mode=SignalMode.BEARING_DEGRADATION) as s:
            r_bearing = s.read()
        k_normal  = float(sp_kurtosis(r_normal.axes["x"],  fisher=True))
        k_bearing = float(sp_kurtosis(r_bearing.axes["x"], fisher=True))
        assert k_bearing > k_normal, \
            f"BEARING kurtosis ({k_bearing:.3f}) debe ser > NORMAL ({k_normal:.3f})"

    def test_11_sensor_failure_detected_as_sensor_error(self):
        """SENSOR_FAILURE: DataQuality lo clasifica como SENSOR_ERROR, no como fallo de máquina."""
        with make_sensor(mode=SignalMode.SENSOR_FAILURE) as s:
            reading = s.read()

        for axis in AXES:
            sig    = reading.axes[axis]
            result = check_signal_quality(sig, FS)
            assert result.is_sensor_error is True, \
                f"Eje {axis}: SENSOR_FAILURE no detectado como SENSOR_ERROR"
            assert result.is_flat is True, \
                f"Eje {axis}: señal plana no detectada"
            assert result.status == "SENSOR_ERROR"

    def test_axes_amplitudes_different(self):
        """Los ejes X/Y/Z deben tener amplitudes distintas (axis_amplitude_factor)."""
        with make_sensor() as s:
            reading = s.read()
        rms_x = float(np.sqrt(np.mean(reading.axes["x"] ** 2)))
        rms_y = float(np.sqrt(np.mean(reading.axes["y"] ** 2)))
        rms_z = float(np.sqrt(np.mean(reading.axes["z"] ** 2)))
        # x > y > z por los factores 1.0, 0.6, 0.3
        assert rms_x > rms_y, f"RMS X ({rms_x:.4f}) debe ser > Y ({rms_y:.4f})"
        assert rms_y > rms_z, f"RMS Y ({rms_y:.4f}) debe ser > Z ({rms_z:.4f})"

    def test_timestamps_monotonic(self):
        """Los timestamps generados deben ser monótonamente crecientes."""
        with make_sensor() as s:
            reading = s.read()
        assert reading.timestamps is not None
        diffs = np.diff(reading.timestamps)
        assert np.all(diffs > 0), "Timestamps no monótonicos en MockSensor"

    def test_reading_count_increments(self):
        """reading_count debe incrementarse en cada lectura."""
        with make_sensor() as s:
            r1 = s.read()
            r2 = s.read()
        assert r1.metadata["reading_count"] == 1
        assert r2.metadata["reading_count"] == 2


# ─── TESTS: PIPELINE DE INTEGRACIÓN ──────────────────────────────────────────

class TestEdgePipeline:
    """Tests de integración: MockSensor → DataQuality → FeatureExtraction."""

    def _run_pipeline(
        self,
        mode:        SignalMode = SignalMode.NORMAL,
        amplitude:   float      = 0.07,
        freq:        float      = 50.0,
        context:     OperatingContext | None = None,
    ) -> tuple[dict, dict, MultiAxisReading]:
        """Helper: ejecuta el pipeline completo para un modo dado."""
        sensor = make_sensor(mode=mode, amplitude=amplitude, freq=freq)
        sensor.configure()
        reading = sensor.read()
        sensor.close()

        quality_per_axis: dict = {}
        valid_signals:    dict = {}

        for axis in AXES:
            sig = reading.axes[axis]
            qr  = check_signal_quality(
                sig, FS,
                timestamps = reading.timestamps,
                odr_hz     = reading.sampling_rate_configured,
            )
            quality_per_axis[axis] = qr
            if qr.is_valid and not qr.is_sensor_error:
                valid_signals[axis] = sig

        features = None
        if valid_signals:
            features = extract_multiaxis_features(
                signals                  = valid_signals,
                sensor_id                = reading.sensor_id,
                sampling_rate_configured = reading.sampling_rate_configured,
                sampling_rate_actual     = reading.sampling_rate_actual,
                context                  = context,
                include_total            = True,
            )

        return quality_per_axis, reading.metadata, features

    def test_1_normal_pipeline_end_to_end(self):
        """Pipeline completo con señal normal: calidad OK y features válidas."""
        qr, meta, features = self._run_pipeline(SignalMode.NORMAL)
        assert all(q.is_valid for q in qr.values())
        assert features is not None
        assert features.x is not None
        assert features.y is not None
        assert features.z is not None
        assert np.isfinite(features.x.time.rms)

    def test_5_sensor_failure_not_classified_as_machine_fault(self):
        """SENSOR_FAILURE: ningún eje debe pasar al procesamiento de features."""
        qr, meta, features = self._run_pipeline(SignalMode.SENSOR_FAILURE)
        # Todos los ejes deben ser SENSOR_ERROR
        for axis, result in qr.items():
            assert result.is_sensor_error is True, \
                f"Eje {axis}: SENSOR_FAILURE no detectado como SENSOR_ERROR"
        # No debe haber features (nada pasó a procesamiento)
        assert features is None, \
            "SENSOR_FAILURE no debe producir features de vibración"

    def test_4_progressive_vibration_rms_increases(self):
        """Señal con amplitud creciente → RMS del eje X crece."""
        rms_values = []
        for amp in [0.05, 0.10, 0.20, 0.40]:
            _, _, features = self._run_pipeline(SignalMode.NORMAL, amplitude=amp)
            assert features is not None
            rms_values.append(features.x.time.rms)

        for i in range(len(rms_values) - 1):
            assert rms_values[i] < rms_values[i + 1], \
                f"RMS no aumenta progresivamente: {rms_values}"

    def test_3_dominant_frequency_in_pipeline(self):
        """Frecuencia dominante correctamente detectada en el pipeline completo."""
        target_hz = 200.0
        _, _, features = self._run_pipeline(SignalMode.NORMAL, freq=target_hz)
        assert features is not None
        # La frecuencia dominante en X debe estar cerca del objetivo
        dom_freq = features.x.freq.dominant_freq
        assert abs(dom_freq - target_hz) < 5.0, \
            f"Esperada {target_hz} Hz, detectada {dom_freq:.1f} Hz"

    def test_rpm_nominal_not_used_as_real(self):
        """Con rpm_nominal pero sin rpm_real, el order analysis no está disponible."""
        ctx = OperatingContext(rpm_nominal=3000.0)
        _, _, features = self._run_pipeline(context=ctx)
        assert features is not None
        assert features.order_prep is not None
        assert features.order_prep.is_available is False
        assert features.order_prep.order_frequency(1) is None

    def test_order_analysis_available_with_rpm_real(self):
        """Con rpm_real, el order analysis está disponible."""
        ctx = OperatingContext(rpm_nominal=3000.0, rpm_real=3000.0, rpm_source="encoder")
        _, _, features = self._run_pipeline(context=ctx)
        assert features is not None
        assert features.order_prep is not None
        assert features.order_prep.is_available is True
        assert abs(features.order_prep.order_frequency(1) - 50.0) < 0.1

    def test_11_all_mock_modes_through_pipeline(self):
        """Todos los modos del MockSensor deben ejecutar sin excepción."""
        modes_with_features = [
            SignalMode.NORMAL,
            SignalMode.IMBALANCE,
            SignalMode.MISALIGNMENT,
            SignalMode.LOOSENESS,
            SignalMode.BEARING_DEGRADATION,
        ]
        for mode in modes_with_features:
            qr, _, features = self._run_pipeline(mode)
            assert features is not None, f"Modo {mode}: sin features"
            assert features.x is not None, f"Modo {mode}: eje X vacío"

        # SENSOR_FAILURE: no debe producir features
        qr, _, features = self._run_pipeline(SignalMode.SENSOR_FAILURE)
        assert features is None, "SENSOR_FAILURE no debe producir features"

    def test_sampling_rate_propagated_to_features(self):
        """sampling_rate_actual debe propagarse al MultiAxisReading."""
        qr, _, features = self._run_pipeline(SignalMode.NORMAL)
        assert features is not None
        assert features.sampling_rate_configured == FS
        # MockSensor siempre reporta actual = configured
        assert features.sampling_rate_actual == FS


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
