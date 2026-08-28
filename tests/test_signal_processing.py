"""
Tests unitarios para src/edge/signal_processing.py

Cubre 5 tipos de señal sintética:
  1. Señal normal (senoide limpia de máquina sana)
  2. Aumento de vibración (señal de alta amplitud)
  3. Señal con frecuencia dominante conocida
  4. Señal con ruido blanco
  5. Señal saturada (clipping)

Cada test verifica propiedades matemáticas conocidas o comportamiento
esperado frente a anomalías de señal.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.edge.signal_processing import (
    # Configuración
    SignalConfig,
    BandDefinition,
    # Dataclasses
    TimeFeatures,
    FrequencyFeatures,
    VibrationFeatures,
    # Preprocesado
    detrend_signal,
    bandpass_filter,
    apply_window,
    validate_signal,
    # Dominio temporal
    compute_rms,
    compute_peak,
    compute_peak_to_peak,
    compute_std,
    compute_kurtosis,
    compute_skewness,
    compute_crest_factor,
    extract_time_features,
    # Dominio frecuencial
    compute_fft,
    dominant_frequency,
    spectral_energy,
    band_energy,
    extract_frequency_features,
    # Pipeline
    process_vibration_signal,
)

# ─── GENERADORES DE SEÑAL SINTÉTICA ────────────────────────────────────────────

FS = 3200.0  # Frecuencia de muestreo estándar para los tests


def gen_normal_signal(fs: float = FS,
                      duration: float = 1.0,
                      freq_hz: float = 50.0,
                      amplitude: float = 0.07) -> np.ndarray:
    """
    Señal 1: Normal — senoide limpia que simula máquina sana.
    Representa vibración de rodamiento en buen estado.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def gen_high_vibration_signal(fs: float = FS,
                               duration: float = 1.0,
                               freq_hz: float = 50.0,
                               amplitude: float = 0.45) -> np.ndarray:
    """
    Señal 2: Alta vibración — senoide de alta amplitud.
    Simula máquina con desgaste severo o desalineación.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def gen_dominant_freq_signal(fs: float = FS,
                              duration: float = 1.0,
                              dominant_hz: float = 200.0) -> np.ndarray:
    """
    Señal 3: Con frecuencia dominante conocida — útil para verificar FFT.
    Mezcla de componentes con el pico claro en dominant_hz.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # Componente dominante (alta amplitud) + armónico menor
    signal = (0.15 * np.sin(2 * np.pi * dominant_hz * t) +
              0.02 * np.sin(2 * np.pi * 50.0 * t) +
              0.01 * np.sin(2 * np.pi * 400.0 * t))
    return signal


def gen_noisy_signal(fs: float = FS,
                     duration: float = 1.0,
                     seed: int = 42) -> np.ndarray:
    """
    Señal 4: Con ruido blanco gaussiano.
    Simula señal real con perturbaciones eléctricas/mecánicas.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    base = 0.07 * np.sin(2 * np.pi * 50.0 * t)
    noise = rng.normal(0, 0.02, size=len(t))
    return base + noise


def gen_saturated_signal(fs: float = FS,
                          duration: float = 1.0,
                          saturation_level: float = 0.5) -> np.ndarray:
    """
    Señal 5: Saturada (clipping) — simula ADC saturado o sensor sobrecargado.
    Valores cortados en ±saturation_level.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    raw = 1.0 * np.sin(2 * np.pi * 50.0 * t)
    return np.clip(raw, -saturation_level, saturation_level)


# ─── TESTS: DETREND ────────────────────────────────────────────────────────────

class TestDetrend:

    def test_removes_dc_offset(self):
        """Un offset DC constante debe eliminarse completamente."""
        signal = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 3200)) + 5.0
        detrended = detrend_signal(signal, detrend_type="constant")
        assert abs(np.mean(detrended)) < 1e-6, \
            "La media debe ser ≈ 0 después de detrend 'constant'"

    def test_removes_linear_trend(self):
        """Una deriva lineal debe eliminarse con detrend 'linear'."""
        t = np.linspace(0, 1, 3200)
        signal = 0.07 * np.sin(2 * np.pi * 50 * t) + 2.0 * t + 1.0  # + rampa
        detrended = detrend_signal(signal, detrend_type="linear")
        # Tras eliminar la tendencia, la media y la pendiente deben ser ≈ 0
        slope = np.polyfit(t, detrended, 1)[0]
        assert abs(slope) < 1e-3, \
            f"Pendiente residual demasiado alta: {slope:.6f}"

    def test_preserves_oscillation(self):
        """La oscilación de la señal debe conservarse."""
        signal = gen_normal_signal()
        detrended = detrend_signal(signal)
        # RMS antes y después deben ser similares (señal ya centrada)
        rms_before = compute_rms(signal)
        rms_after  = compute_rms(detrended)
        assert abs(rms_before - rms_after) / rms_before < 0.01, \
            "El RMS no debe cambiar más de un 1% en señal ya centrada"

    def test_rejects_short_signal(self):
        """Señales de menos de 2 muestras deben lanzar ValueError."""
        with pytest.raises(ValueError, match="al menos 2"):
            detrend_signal(np.array([1.0]))

    def test_rejects_invalid_type(self):
        """Tipo de detrend inválido debe lanzar ValueError."""
        with pytest.raises(ValueError, match="detrend_type"):
            detrend_signal(np.ones(100), detrend_type="cubic")


# ─── TESTS: BANDPASS FILTER ────────────────────────────────────────────────────

class TestBandpassFilter:

    def test_attenuates_very_low_frequency(self):
        """Frecuencias por debajo del corte deben atenuarse."""
        fs = FS
        t  = np.linspace(0, 2, int(fs * 2), endpoint=False)
        # Señal de 2 Hz (muy por debajo del corte en 10 Hz)
        low_freq_signal = np.sin(2 * np.pi * 2 * t)
        filtered = bandpass_filter(low_freq_signal, fs,
                                   low_hz=10.0, high_hz=800.0)
        # La energía filtrada debe ser << energía original
        energy_ratio = compute_rms(filtered) / compute_rms(low_freq_signal)
        assert energy_ratio < 0.1, \
            f"Señal de 2 Hz no atenuada suficientemente: ratio={energy_ratio:.3f}"

    def test_passes_in_band_frequency(self):
        """Frecuencias dentro de la banda deben pasar con poca atenuación."""
        fs = FS
        t  = np.linspace(0, 2, int(fs * 2), endpoint=False)
        # Señal de 100 Hz (centro de la banda 10–800 Hz)
        in_band = np.sin(2 * np.pi * 100 * t)
        filtered = bandpass_filter(in_band, fs,
                                   low_hz=10.0, high_hz=800.0)
        energy_ratio = compute_rms(filtered) / compute_rms(in_band)
        assert energy_ratio > 0.8, \
            f"Señal de 100 Hz atenuada demasiado: ratio={energy_ratio:.3f}"

    def test_rejects_invalid_high_freq(self):
        """high_hz >= Nyquist debe lanzar ValueError."""
        with pytest.raises(ValueError, match="Nyquist"):
            bandpass_filter(np.ones(3200), fs=3200.0,
                            low_hz=10.0, high_hz=1600.0)  # = Nyquist

    def test_rejects_low_ge_high(self):
        """low_hz >= high_hz debe lanzar ValueError."""
        with pytest.raises(ValueError, match="low_hz.*high_hz"):
            bandpass_filter(np.ones(3200), fs=3200.0,
                            low_hz=500.0, high_hz=100.0)


# ─── TESTS: WINDOWING ──────────────────────────────────────────────────────────

class TestWindowing:

    def test_hann_zero_at_edges(self):
        """La ventana Hann debe ser ≈ 0 en los bordes."""
        signal    = np.ones(1024)
        windowed  = apply_window(signal, "hann")
        assert windowed[0]  < 1e-6, "Borde inicial de Hann debe ser ≈ 0"
        assert windowed[-1] < 1e-6, "Borde final de Hann debe ser ≈ 0"

    def test_rectangular_preserves_signal(self):
        """La ventana rectangular no debe modificar la señal."""
        signal   = gen_normal_signal()
        windowed = apply_window(signal, "rectangular")
        np.testing.assert_array_equal(signal, windowed)

    def test_all_windows_same_length(self):
        """Todas las ventanas soportadas deben devolver la misma longitud."""
        signal   = gen_normal_signal()
        for wtype in ["hann", "hamming", "blackman", "flattop", "rectangular"]:
            windowed = apply_window(signal, wtype)
            assert len(windowed) == len(signal), \
                f"Ventana '{wtype}' cambió la longitud de la señal"

    def test_rejects_unknown_window(self):
        """Tipo de ventana desconocido debe lanzar ValueError."""
        with pytest.raises(ValueError, match="no soportada"):
            apply_window(np.ones(100), "kaiser")


# ─── TESTS: VALIDATE SIGNAL ────────────────────────────────────────────────────

class TestValidateSignal:

    def test_valid_normal_signal(self):
        """Señal normal debe ser válida con calidad alta."""
        result = validate_signal(gen_normal_signal(), FS)
        assert result["is_valid"] is True
        assert result["quality_score"] > 0.8

    def test_flat_signal_invalid(self):
        """Señal plana (sensor desconectado) debe ser inválida."""
        flat   = np.zeros(3200)
        result = validate_signal(flat, FS)
        assert result["is_valid"] is False
        assert any("plana" in w.lower() for w in result["warnings"])

    def test_nan_signal_invalid(self):
        """Señal con NaN debe ser inválida."""
        signal = gen_normal_signal()
        signal[100] = np.nan
        result = validate_signal(signal, FS)
        assert result["is_valid"] is False

    def test_saturated_signal_warning(self):
        """Señal saturada debe generar advertencia."""
        # Señal completamente saturada (todos los valores = ±0.5)
        saturated = np.full(3200, 0.5)
        saturated[::2] = -0.5
        result = validate_signal(saturated, FS)
        # Puede ser inválida o tener advertencia, pero no debe pasar silencioso
        has_warning = len(result["warnings"]) > 0
        low_quality = result["quality_score"] < 0.8
        assert has_warning or low_quality or not result["is_valid"]

    def test_short_signal_invalid(self):
        """Señal muy corta debe ser inválida."""
        result = validate_signal(np.array([0.1, 0.2, -0.1]), FS)
        assert result["is_valid"] is False


# ─── TESTS: FEATURES TEMPORALES ────────────────────────────────────────────────

class TestTimeFeatures:

    def test_rms_sinusoid_analytical(self):
        """RMS de A·sin(ωt) = A/sqrt(2) — resultado analítico exacto."""
        amplitude = 0.07
        signal    = amplitude * np.sin(
            2 * np.pi * 50 * np.linspace(0, 10, 32000, endpoint=False)
        )
        expected_rms = amplitude / np.sqrt(2)
        computed_rms = compute_rms(signal)
        assert abs(computed_rms - expected_rms) < 1e-4, \
            f"RMS esperado {expected_rms:.6f}, calculado {computed_rms:.6f}"

    def test_peak_positive(self):
        """El Peak siempre debe ser positivo."""
        for signal in [gen_normal_signal(), gen_noisy_signal(), gen_saturated_signal()]:
            assert compute_peak(signal) > 0

    def test_peak_to_peak_ge_peak(self):
        """Peak-to-Peak siempre >= Peak (para señal simétrica ≈ 2×Peak)."""
        signal = gen_normal_signal()
        assert compute_peak_to_peak(signal) >= compute_peak(signal)

    def test_std_zero_constant_signal(self):
        """Desviación estándar de señal constante debe ser 0."""
        constant = np.full(3200, 1.5)
        assert compute_std(constant) < 1e-10

    def test_kurtosis_gaussian_near_zero(self):
        """Señal gaussiana debe tener kurtosis (Fisher) ≈ 0."""
        rng    = np.random.default_rng(42)
        gauss  = rng.normal(0, 1, 100000)
        kurt   = compute_kurtosis(gauss)
        assert abs(kurt) < 0.1, \
            f"Kurtosis gaussiana esperada ≈ 0, calculada {kurt:.4f}"

    def test_kurtosis_impulsive_signal_high(self):
        """Señal con impactos debe tener kurtosis positiva y alta."""
        rng     = np.random.default_rng(42)
        # Señal base + impulsos esparcidos
        base    = rng.normal(0, 0.01, 10000)
        impulse_idx = rng.integers(0, 10000, 50)
        base[impulse_idx] = rng.choice([-1, 1], 50) * 1.0
        kurt = compute_kurtosis(base)
        assert kurt > 3.0, \
            f"Señal impulsiva debe tener kurtosis > 3, obtenido {kurt:.4f}"

    def test_skewness_symmetric_near_zero(self):
        """Señal simétrica debe tener skewness ≈ 0."""
        signal = gen_normal_signal()
        skew   = compute_skewness(signal)
        assert abs(skew) < 0.05, \
            f"Señal simétrica debe tener skewness ≈ 0, obtenido {skew:.6f}"

    def test_crest_factor_sinusoid_analytical(self):
        """Crest Factor de senoide pura = sqrt(2) ≈ 1.414."""
        signal    = gen_normal_signal(duration=10.0)  # muchos ciclos
        cf        = compute_crest_factor(signal)
        expected  = np.sqrt(2)
        assert abs(cf - expected) < 0.01, \
            f"Crest Factor esperado {expected:.4f}, calculado {cf:.4f}"

    def test_crest_factor_zero_signal(self):
        """Señal de ceros debe devolver crest_factor=0.0 sin error."""
        signal = np.zeros(3200)
        assert compute_crest_factor(signal) == 0.0

    def test_extract_time_features_returns_all(self):
        """extract_time_features debe devolver un TimeFeatures completo."""
        signal = gen_normal_signal()
        feats  = extract_time_features(signal)
        assert isinstance(feats, TimeFeatures)
        assert all(np.isfinite(v) for v in [
            feats.rms, feats.peak, feats.peak_to_peak,
            feats.std, feats.kurtosis, feats.skewness, feats.crest_factor
        ])

    def test_high_vibration_rms_higher(self):
        """Señal 2: alta vibración debe tener RMS mayor que señal normal."""
        normal_rms = compute_rms(gen_normal_signal())
        high_rms   = compute_rms(gen_high_vibration_signal())
        assert high_rms > normal_rms * 3, \
            f"RMS alta vibración ({high_rms:.4f}) no es suficientemente mayor " \
            f"que normal ({normal_rms:.4f})"


# ─── TESTS: FFT Y FEATURES FRECUENCIALES ───────────────────────────────────────

class TestFFTFeatures:

    def test_fft_dominant_freq_detected(self):
        """Señal 3: la frecuencia dominante conocida debe detectarse correctamente."""
        target_hz = 200.0
        signal    = gen_dominant_freq_signal(dominant_hz=target_hz)
        freqs, mags = compute_fft(signal, FS)
        dom_freq, _ = dominant_frequency(freqs, mags)
        # Tolerancia de ±2 Hz (resolución espectral ≈ fs/N)
        assert abs(dom_freq - target_hz) < 2.0, \
            f"Frecuencia dominante esperada {target_hz} Hz, " \
            f"detectada {dom_freq:.1f} Hz"

    def test_fft_frequencies_positive(self):
        """Todas las frecuencias FFT deben ser ≥ 0 (rfft)."""
        signal = gen_normal_signal()
        freqs, _ = compute_fft(signal, FS)
        assert np.all(freqs >= 0)

    def test_fft_max_freq_is_nyquist(self):
        """La frecuencia máxima en el array debe ser ≤ Nyquist."""
        signal = gen_normal_signal()
        freqs, _ = compute_fft(signal, FS)
        assert freqs[-1] <= FS / 2 + 1e-6

    def test_spectral_energy_positive(self):
        """La energía espectral debe ser siempre positiva."""
        for signal in [gen_normal_signal(), gen_noisy_signal(),
                       gen_high_vibration_signal()]:
            _, mags = compute_fft(signal, FS)
            assert spectral_energy(mags) > 0

    def test_band_energy_within_full_energy(self):
        """La energía de una banda debe ser ≤ energía total."""
        signal = gen_noisy_signal()
        freqs, mags = compute_fft(signal, FS)
        total = spectral_energy(mags)
        low_band = band_energy(freqs, mags, 10, 100)
        assert low_band <= total * 1.001  # pequeño margen numérico

    def test_band_energy_empty_band_zero(self):
        """Banda fuera del rango de la señal debe devolver 0."""
        signal = gen_normal_signal()
        freqs, mags = compute_fft(signal, FS)
        # Banda entre 0 y -10 Hz (imposible, sin frecuencias)
        energy = band_energy(freqs, mags, f_low=2000, f_high=2100)
        assert energy == 0.0

    def test_noisy_signal_spectral_energy_spreads(self):
        """Señal 4: con ruido debe tener energía repartida en más bandas."""
        config = SignalConfig(fs=FS)
        clean  = gen_normal_signal()
        noisy  = gen_noisy_signal()

        ff_clean = extract_frequency_features(clean, FS, config)
        ff_noisy = extract_frequency_features(noisy, FS, config)

        # La energía en la banda alta debe ser mayor para señal ruidosa
        high_clean = ff_clean.band_energies.get("high", 0)
        high_noisy = ff_noisy.band_energies.get("high", 0)
        assert high_noisy > high_clean, \
            "La señal ruidosa debe tener más energía en alta frecuencia"


# ─── TESTS: PIPELINE COMPLETO ──────────────────────────────────────────────────

class TestFullPipeline:

    def test_normal_signal_pipeline(self):
        """Señal 1: el pipeline completo debe completarse sin errores."""
        signal   = gen_normal_signal()
        features = process_vibration_signal(signal, FS)

        assert isinstance(features, VibrationFeatures)
        assert features.signal_length == len(signal)
        assert features.fs == FS
        assert np.isfinite(features.RMS)
        assert np.isfinite(features.Kurtosis)

    def test_high_vibration_detected(self):
        """Señal 2: alta vibración debe producir RMS significativamente mayor."""
        normal_feat = process_vibration_signal(gen_normal_signal(), FS)
        high_feat   = process_vibration_signal(gen_high_vibration_signal(), FS)

        assert high_feat.RMS > normal_feat.RMS * 3, \
            "Alta vibración debe producir RMS al menos 3× mayor"

    def test_dominant_frequency_pipeline(self):
        """Señal 3: la frecuencia dominante debe identificarse en el pipeline."""
        target_hz = 150.0
        signal    = gen_dominant_freq_signal(dominant_hz=target_hz)
        features  = process_vibration_signal(signal, FS)

        dom_freq = features.frequency.dominant_freq
        assert abs(dom_freq - target_hz) < 5.0, \
            f"Pipeline: frecuencia dominante esperada {target_hz} Hz, " \
            f"detectada {dom_freq:.1f} Hz"

    def test_noisy_signal_pipeline(self):
        """Señal 4: señal con ruido debe procesarse sin errores."""
        signal   = gen_noisy_signal()
        features = process_vibration_signal(signal, FS)
        assert isinstance(features, VibrationFeatures)
        # El ruido debe aumentar la energía en bandas altas
        assert features.frequency.band_energies.get("high", 0) > 0

    def test_saturated_signal_raises_or_warns(self):
        """Señal completamente plana (std=0) debe fallar validación con validate=True."""
        # Señal de valor constante: std=0 → validate_signal devuelve is_valid=False
        flat_signal = np.ones(3200)
        with pytest.raises((ValueError, Exception)):
            process_vibration_signal(flat_signal, FS, validate=True)

    def test_saturated_signal_without_validation(self):
        """Señal 5: con validate=False el pipeline debe completarse."""
        signal   = gen_saturated_signal()
        features = process_vibration_signal(signal, FS, validate=False)
        assert isinstance(features, VibrationFeatures)

    def test_api_dict_compatibility(self):
        """La salida debe ser compatible con el endpoint POST /predict/bearing."""
        signal   = gen_normal_signal()
        features = process_vibration_signal(signal, FS)
        api_dict = features.to_api_dict(maquina="Torno_CNC_1")

        required_keys = {"maquina", "RMS", "Peak_to_Peak", "Kurtosis", "Skewness"}
        assert required_keys.issubset(api_dict.keys()), \
            f"Faltan claves en to_api_dict: {required_keys - api_dict.keys()}"
        assert api_dict["maquina"] == "Torno_CNC_1"
        assert all(isinstance(api_dict[k], float) for k in
                   ["RMS", "Peak_to_Peak", "Kurtosis", "Skewness"])

    def test_custom_config(self):
        """El pipeline debe funcionar con configuración personalizada."""
        config = SignalConfig(
            fs=FS,
            window_type="hamming",
            bandpass_low_hz=20.0,
            bandpass_high_hz=800.0,
            bands=[
                BandDefinition("very_low", 20, 50),
                BandDefinition("medium",   50, 200),
            ]
        )
        signal   = gen_normal_signal()
        features = process_vibration_signal(signal, FS, config=config)

        assert "very_low" in features.frequency.band_energies
        assert "medium"   in features.frequency.band_energies

    def test_output_all_finite(self):
        """Todas las features de salida deben ser números finitos."""
        for signal in [gen_normal_signal(), gen_high_vibration_signal(),
                       gen_dominant_freq_signal(), gen_noisy_signal()]:
            features = process_vibration_signal(signal, FS, validate=False)
            assert np.isfinite(features.time.rms)
            assert np.isfinite(features.time.kurtosis)
            assert np.isfinite(features.time.crest_factor)
            assert np.isfinite(features.frequency.dominant_freq)
            assert np.isfinite(features.frequency.spectral_energy)

    def test_compatible_properties(self):
        """Las propiedades de compatibilidad deben coincidir con time features."""
        signal   = gen_normal_signal()
        features = process_vibration_signal(signal, FS)

        assert features.RMS          == features.time.rms
        assert features.Peak_to_Peak == features.time.peak_to_peak
        assert features.Kurtosis     == features.time.kurtosis
        assert features.Skewness     == features.time.skewness


# ─── TESTS DE REGRESIÓN ────────────────────────────────────────────────────────

class TestRegressionValues:
    """
    Verifica valores específicos contra resultados conocidos.
    Detecta regresiones si se modifica el algoritmo.
    """

    def test_rms_known_value(self):
        """RMS de 1.0·sin(2π·50·t) debe ser exactamente 1/sqrt(2)."""
        t    = np.linspace(0, 100, 320000, endpoint=False)  # 100s, 3200 Hz
        sig  = 1.0 * np.sin(2 * np.pi * 50 * t)
        rms  = compute_rms(sig)
        assert abs(rms - 1.0 / np.sqrt(2)) < 1e-5

    def test_peak_to_peak_known_value(self):
        """P2P de señal entre -0.3 y +0.3 debe ser 0.6."""
        t   = np.linspace(0, 1, 3200, endpoint=False)
        sig = 0.3 * np.sin(2 * np.pi * 50 * t)
        p2p = compute_peak_to_peak(sig)
        assert abs(p2p - 0.6) < 0.01

    def test_crest_factor_known_value(self):
        """CF de señal cuadrada perfecta = 1.0."""
        square = np.array([1.0, -1.0] * 1600, dtype=float)
        cf = compute_crest_factor(square)
        assert abs(cf - 1.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
