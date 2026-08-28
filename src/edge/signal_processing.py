"""
AuraPredict — Signal Processing Module
=======================================
Procesamiento matemático puro de señales de vibración industrial.

Completamente hardware-agnostic: recibe un array NumPy de aceleración
y devuelve features. Funciona con datos procedentes de:
  - ADXL345 (MEMS, I2C)
  - Acelerómetros industriales IEPE
  - Cualquier otro MEMS
  - Simuladores y datos sintéticos

No importa ni usa ningún módulo de hardware.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew
from dataclasses import dataclass, field
from typing import Optional


# ─── CONFIGURACIÓN ─────────────────────────────────────────────────────────────

@dataclass
class BandDefinition:
    """Define un rango de frecuencias para análisis de energía."""
    name: str
    low_hz: float
    high_hz: float


@dataclass
class SignalConfig:
    """
    Configuración del pipeline de procesamiento.

    Parámetros ajustables por máquina sin tocar el código.
    """
    fs: float = 3200.0              # Frecuencia de muestreo (Hz)
    window_type: str = "hann"       # Tipo de ventana para FFT
    filter_order: int = 4           # Orden del filtro Butterworth
    bandpass_low_hz: float = 10.0   # Frecuencia de corte inferior (Hz)
    bandpass_high_hz: float = 1000.0  # Frecuencia de corte superior (Hz)
    detrend_type: str = "linear"    # 'linear' | 'constant'

    # Bandas de energía configurables
    bands: list[BandDefinition] = field(default_factory=lambda: [
        BandDefinition("low",  10.0,  100.0),
        BandDefinition("mid",  100.0, 500.0),
        BandDefinition("high", 500.0, 1600.0),
    ])

    def __post_init__(self):
        nyquist = self.fs / 2.0
        if self.bandpass_high_hz >= nyquist:
            # Limitar al 95% de Nyquist para evitar inestabilidad del filtro
            self.bandpass_high_hz = nyquist * 0.95


# ─── DATACLASSES DE SALIDA ─────────────────────────────────────────────────────

@dataclass
class TimeFeatures:
    """
    Features extraídas en el dominio temporal.

    Kurtosis: definición Fisher (exceso), valor 0 para señal gaussiana.
    Skewness: definición estándar, 0 para distribución simétrica.
    Crest Factor: Peak / RMS — indicador temprano de fallos impulsivos.
    """
    rms:           float   # Root Mean Square — energía global
    peak:          float   # Valor máximo de amplitud absoluta
    peak_to_peak:  float   # Rango pico a pico (max - min)
    std:           float   # Desviación estándar
    kurtosis:      float   # Impulsividad de la señal (Fisher, normal=0)
    skewness:      float   # Asimetría de la distribución
    crest_factor:  float   # Peak / RMS (normal ≈ 1.4 para senoide)


@dataclass
class FrequencyFeatures:
    """Features extraídas en el dominio de la frecuencia."""
    dominant_freq:      float               # Frecuencia dominante (Hz)
    dominant_amplitude: float               # Amplitud de la frecuencia dominante
    spectral_energy:    float               # Energía espectral total
    band_energies:      dict[str, float]    # Energía por banda {name: value}
    freqs:              np.ndarray          # Array de frecuencias FFT
    magnitudes:         np.ndarray          # Array de magnitudes FFT


@dataclass
class VibrationFeatures:
    """
    Conjunto completo de features de una señal de vibración procesada.

    Incluye propiedades de compatibilidad con el sistema existente
    (API, diagnóstico, BD) para no romper nada durante la migración.
    """
    time:          TimeFeatures
    frequency:     FrequencyFeatures
    signal_length: int
    fs:            float
    config:        SignalConfig

    # ── Compatibilidad con sistema existente ──────────────────────────────
    @property
    def RMS(self) -> float:
        return self.time.rms

    @property
    def Peak_to_Peak(self) -> float:
        return self.time.peak_to_peak

    @property
    def Kurtosis(self) -> float:
        return self.time.kurtosis

    @property
    def Skewness(self) -> float:
        return self.time.skewness

    def to_api_dict(self, maquina: str = "CNC_1") -> dict:
        """
        Devuelve el diccionario compatible con el endpoint
        POST /predict/bearing de la API actual.
        """
        return {
            "maquina":      maquina,
            "RMS":          round(self.time.rms, 4),
            "Peak_to_Peak": round(self.time.peak_to_peak, 4),
            "Kurtosis":     round(self.time.kurtosis, 4),
            "Skewness":     round(self.time.skewness, 4),
        }


# ─── PREPROCESADO ──────────────────────────────────────────────────────────────

def detrend_signal(signal: np.ndarray,
                   detrend_type: str = "linear") -> np.ndarray:
    """
    Elimina la tendencia de la señal.

    'constant' → elimina offset DC (media).
    'linear'   → elimina deriva lineal + DC (recomendado para vibración).

    Args:
        signal: Array 1D de muestras de aceleración.
        detrend_type: 'linear' | 'constant'.

    Returns:
        Señal sin tendencia.
    """
    if signal.ndim != 1:
        raise ValueError(f"Se esperaba señal 1D, se recibió shape {signal.shape}")
    if len(signal) < 2:
        raise ValueError("La señal debe tener al menos 2 muestras")

    detrend_map = {"linear": "linear", "constant": "constant"}
    if detrend_type not in detrend_map:
        raise ValueError(f"detrend_type debe ser 'linear' o 'constant', "
                         f"se recibió '{detrend_type}'")

    return sp_signal.detrend(signal, type=detrend_map[detrend_type])


def bandpass_filter(signal: np.ndarray,
                    fs: float,
                    low_hz: float,
                    high_hz: float,
                    order: int = 4) -> np.ndarray:
    """
    Filtro Butterworth paso de banda.

    Elimina componentes de muy baja frecuencia (movimiento rígido de
    cuerpo) y de muy alta frecuencia (ruido eléctrico fuera de banda).

    Args:
        signal:  Array 1D de muestras.
        fs:      Frecuencia de muestreo (Hz).
        low_hz:  Frecuencia de corte inferior (Hz).
        high_hz: Frecuencia de corte superior (Hz).
        order:   Orden del filtro (4 = buen compromiso atenuación/estabilidad).

    Returns:
        Señal filtrada.

    Raises:
        ValueError: Si los parámetros de frecuencia son inválidos.
    """
    nyquist = fs / 2.0

    if low_hz <= 0:
        raise ValueError(f"low_hz debe ser > 0, se recibió {low_hz}")
    if high_hz >= nyquist:
        raise ValueError(
            f"high_hz ({high_hz} Hz) debe ser < Nyquist ({nyquist} Hz). "
            f"Reduce bandpass_high_hz o aumenta la frecuencia de muestreo."
        )
    if low_hz >= high_hz:
        raise ValueError(
            f"low_hz ({low_hz}) debe ser < high_hz ({high_hz})"
        )

    low_norm  = low_hz  / nyquist
    high_norm = high_hz / nyquist

    b, a = sp_signal.butter(order, [low_norm, high_norm], btype="band")

    # sosfilt es más estable numéricamente que lfilter para órdenes altos
    sos = sp_signal.butter(order, [low_norm, high_norm],
                           btype="band", output="sos")
    return sp_signal.sosfilt(sos, signal)


def apply_window(signal: np.ndarray,
                 window_type: str = "hann") -> np.ndarray:
    """
    Aplica una función de ventana a la señal antes de la FFT.

    Reduce el leakage espectral causado por la discontinuidad en los
    bordes de la ventana de análisis.

    Ventanas soportadas: hann, hamming, blackman, flattop, rectangular.

    Args:
        signal:      Array 1D de muestras.
        window_type: Tipo de ventana.

    Returns:
        Señal multiplicada por la ventana.
    """
    n = len(signal)
    windows = {
        "hann":        np.hanning(n),
        "hamming":     np.hamming(n),
        "blackman":    np.blackman(n),
        "flattop":     sp_signal.windows.flattop(n),
        "rectangular": np.ones(n),
    }
    if window_type not in windows:
        raise ValueError(
            f"Ventana '{window_type}' no soportada. "
            f"Opciones: {list(windows.keys())}"
        )
    return signal * windows[window_type]


def validate_signal(signal: np.ndarray,
                    fs: float,
                    saturation_limit: float = 0.99) -> dict:
    """
    Valida la calidad de una señal antes de procesarla.

    Detecta señales inválidas que producirían features sin sentido:
    señal plana (sensor desconectado), señal saturada (clipping),
    señal demasiado corta.

    Args:
        signal:           Array 1D de muestras.
        fs:               Frecuencia de muestreo (Hz).
        saturation_limit: Fracción de muestras saturadas que dispara alerta.

    Returns:
        Diccionario con:
          - quality_score (0–1): 1 = señal perfecta
          - warnings (list[str]): lista de problemas detectados
          - is_valid (bool): False si la señal no debe procesarse
    """
    warnings = []
    quality_score = 1.0

    if signal.ndim != 1 or len(signal) < 32:
        return {"quality_score": 0.0, "warnings": ["Señal demasiado corta"],
                "is_valid": False}

    # Señal plana (sensor desconectado o cero)
    std = float(np.std(signal))
    if std < 1e-9:
        return {"quality_score": 0.0,
                "warnings": ["Señal plana — posible sensor desconectado"],
                "is_valid": False}

    # Señal saturada (clipping)
    max_abs = float(np.max(np.abs(signal)))
    if max_abs > 0:
        saturated_fraction = float(
            np.sum(np.abs(signal) >= max_abs * 0.999) / len(signal)
        )
        if saturated_fraction > saturation_limit:
            warnings.append(
                f"Señal saturada — {saturated_fraction*100:.1f}% de muestras "
                f"en el límite ({max_abs:.3f})"
            )
            quality_score -= 0.5

    # NaN o infinitos
    if not np.all(np.isfinite(signal)):
        return {"quality_score": 0.0,
                "warnings": ["Señal contiene NaN o infinito"],
                "is_valid": False}

    # SNR aproximado (std vs rango)
    dynamic_range = max_abs
    noise_floor = std / dynamic_range if dynamic_range > 0 else 0
    if noise_floor < 0.001:
        warnings.append("Posible señal muy ruidosa o rango dinámico bajo")
        quality_score -= 0.2

    quality_score = max(0.0, min(1.0, quality_score))
    return {
        "quality_score": round(quality_score, 3),
        "warnings":      warnings,
        "is_valid":      quality_score > 0.3,
    }


# ─── DOMINIO TEMPORAL ──────────────────────────────────────────────────────────

def compute_rms(signal: np.ndarray) -> float:
    """
    Root Mean Square — energía global de la vibración.

    El RMS cuantifica la energía total de la señal.
    Un incremento sostenido indica degradación generalizada.
    Para señal senoidal pura: RMS = Amplitud / sqrt(2).

    Returns:
        Valor RMS (misma unidad que la señal de entrada, p.ej. g o m/s²).
    """
    return float(np.sqrt(np.mean(signal ** 2)))


def compute_peak(signal: np.ndarray) -> float:
    """
    Valor de pico — máxima amplitud absoluta de la señal.

    Detecta impactos puntuales de alta energía.

    Returns:
        Máximo valor absoluto de la señal.
    """
    return float(np.max(np.abs(signal)))


def compute_peak_to_peak(signal: np.ndarray) -> float:
    """
    Amplitud pico a pico — diferencia entre máximo y mínimo.

    Especialmente sensible a impactos asimétricos y holguras mecánicas.

    Returns:
        max(signal) - min(signal).
    """
    return float(np.max(signal) - np.min(signal))


def compute_std(signal: np.ndarray) -> float:
    """
    Desviación estándar de la señal.

    Para señal sin media (detrended), std ≈ RMS.
    La diferencia entre std y RMS indica offset DC residual.

    Returns:
        Desviación estándar.
    """
    return float(np.std(signal))


def compute_kurtosis(signal: np.ndarray) -> float:
    """
    Kurtosis (definición Fisher, exceso de curtosis).

    Mide la impulsividad de la señal.
    - Señal gaussiana: ≈ 0
    - Señal con impactos (rodamiento dañado): > 1
    - Fallo incipiente: típicamente 1–3
    - Fallo claro: típicamente 3–6
    - Fallo severo: > 6

    Nota: el sistema legado usa valores sintéticos (0.62 sano, 5.2 fallo)
    que no corresponden a una definición estadística estándar.
    Este módulo usa la definición Fisher correcta.

    Returns:
        Exceso de kurtosis (Fisher). Positivo = más impulsivo que gaussiana.
    """
    return float(sp_kurtosis(signal, fisher=True, bias=False))


def compute_skewness(signal: np.ndarray) -> float:
    """
    Asimetría (Skewness) de la distribución de amplitudes.

    Valores distintos de 0 indican asimetría en la señal de vibración,
    asociada a desalineación o holgura mecánica.
    - Señal simétrica: ≈ 0
    - Asimetría positiva: impactos hacia valores positivos
    - Asimetría negativa: impactos hacia valores negativos

    Returns:
        Coeficiente de asimetría (dimensionless).
    """
    return float(sp_skew(signal, bias=False))


def compute_crest_factor(signal: np.ndarray) -> float:
    """
    Factor de cresta — Peak / RMS.

    Indicador temprano de fallo: sube antes que el RMS cuando aparecen
    impactos, porque el numerador (Peak) reacciona antes.

    - Señal senoidal pura: sqrt(2) ≈ 1.414
    - Señal sana de rodamiento: 2–4
    - Fallo incipiente: 4–6
    - Fallo avanzado: puede BAJAR (el RMS sube más que el Peak en fallos graves)

    Returns:
        Valor adimensional. Devuelve 0.0 si RMS = 0.
    """
    rms_val = compute_rms(signal)
    if rms_val < 1e-12:
        return 0.0
    return float(compute_peak(signal) / rms_val)


def extract_time_features(signal: np.ndarray) -> TimeFeatures:
    """
    Extrae todas las features del dominio temporal en un solo paso.

    Args:
        signal: Array 1D de muestras (preferiblemente ya detrended).

    Returns:
        TimeFeatures con todas las métricas calculadas.
    """
    return TimeFeatures(
        rms          = compute_rms(signal),
        peak         = compute_peak(signal),
        peak_to_peak = compute_peak_to_peak(signal),
        std          = compute_std(signal),
        kurtosis     = compute_kurtosis(signal),
        skewness     = compute_skewness(signal),
        crest_factor = compute_crest_factor(signal),
    )


# ─── DOMINIO DE FRECUENCIA ─────────────────────────────────────────────────────

def compute_fft(signal: np.ndarray,
                fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula la FFT de la señal.

    Usa rfft (FFT real) que es más eficiente para señales reales
    y devuelve solo la mitad positiva del espectro.
    Las magnitudes se normalizan para representar amplitudes reales.

    Args:
        signal: Array 1D de muestras (ya windowed).
        fs:     Frecuencia de muestreo (Hz).

    Returns:
        Tupla (freqs, magnitudes):
          - freqs:      Array de frecuencias (Hz), rango [0, fs/2].
          - magnitudes: Array de amplitudes (misma unidad que señal).
    """
    n = len(signal)
    freqs      = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals   = np.fft.rfft(signal)
    magnitudes = (2.0 / n) * np.abs(fft_vals)
    # El bin DC (freqs[0]) no se duplica
    if len(magnitudes) > 0:
        magnitudes[0] /= 2.0
    return freqs, magnitudes


def dominant_frequency(freqs: np.ndarray,
                       magnitudes: np.ndarray,
                       min_freq_hz: float = 1.0) -> tuple[float, float]:
    """
    Encuentra la frecuencia dominante y su amplitud.

    Ignora el bin DC y frecuencias muy bajas (< min_freq_hz)
    para evitar artefactos de offset.

    Args:
        freqs:       Array de frecuencias (Hz).
        magnitudes:  Array de magnitudes.
        min_freq_hz: Frecuencia mínima a considerar (Hz).

    Returns:
        Tupla (freq_hz, amplitude).
    """
    # Enmascarar frecuencias muy bajas
    mask = freqs >= min_freq_hz
    if not np.any(mask):
        return 0.0, 0.0

    filtered_mags  = magnitudes.copy()
    filtered_mags[~mask] = 0.0

    idx_max  = int(np.argmax(filtered_mags))
    freq_dom = float(freqs[idx_max])
    amp_dom  = float(magnitudes[idx_max])
    return freq_dom, amp_dom


def spectral_energy(magnitudes: np.ndarray) -> float:
    """
    Energía espectral total (suma de magnitudes al cuadrado).

    Basado en el teorema de Parseval: la energía en el dominio
    del tiempo es igual a la energía en el dominio de la frecuencia.

    Args:
        magnitudes: Array de magnitudes FFT.

    Returns:
        Energía espectral total (adimensional, proporcional a energía real).
    """
    return float(np.sum(magnitudes ** 2))


def band_energy(freqs: np.ndarray,
                magnitudes: np.ndarray,
                f_low: float,
                f_high: float) -> float:
    """
    Energía en una banda de frecuencias específica.

    Útil para monitorizar bandas características de fallos:
    - 1× RPM: desequilibrio
    - 2× RPM: desalineación
    - BPFO, BPFI: fallos de rodamiento

    Args:
        freqs:      Array de frecuencias (Hz).
        magnitudes: Array de magnitudes.
        f_low:      Límite inferior de la banda (Hz).
        f_high:     Límite superior de la banda (Hz).

    Returns:
        Energía en la banda especificada.
    """
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return 0.0
    return float(np.sum(magnitudes[mask] ** 2))


def extract_frequency_features(signal: np.ndarray,
                                fs: float,
                                config: SignalConfig) -> FrequencyFeatures:
    """
    Extrae todas las features del dominio de la frecuencia.

    Pasos:
      1. Aplicar ventana (reduce leakage espectral)
      2. Calcular FFT
      3. Encontrar frecuencia dominante
      4. Calcular energía total
      5. Calcular energía por bandas

    Args:
        signal: Array 1D (detrended y filtrado).
        fs:     Frecuencia de muestreo (Hz).
        config: Configuración del procesado.

    Returns:
        FrequencyFeatures completo.
    """
    # Windowing → FFT
    windowed   = apply_window(signal, config.window_type)
    freqs, mags = compute_fft(windowed, fs)

    # Frecuencia dominante
    dom_freq, dom_amp = dominant_frequency(freqs, mags)

    # Energía por bandas configurables
    b_energies = {}
    for band in config.bands:
        b_energies[band.name] = band_energy(freqs, mags,
                                             band.low_hz, band.high_hz)

    return FrequencyFeatures(
        dominant_freq      = dom_freq,
        dominant_amplitude = dom_amp,
        spectral_energy    = spectral_energy(mags),
        band_energies      = b_energies,
        freqs              = freqs,
        magnitudes         = mags,
    )


# ─── PIPELINE COMPLETO ─────────────────────────────────────────────────────────

def process_vibration_signal(
    raw_signal: np.ndarray,
    fs: float,
    config: Optional[SignalConfig] = None,
    validate: bool = True,
) -> VibrationFeatures:
    """
    Pipeline completo de procesamiento de señal de vibración.

    Pasos en orden:
      1. Validar calidad de la señal (opcional)
      2. Detrend (eliminar offset DC y deriva lineal)
      3. Filtro paso de banda (eliminar fuera de rango útil)
      4. Extraer features temporales (RMS, Kurtosis, Crest...)
      5. Aplicar ventana (Hann por defecto)
      6. Calcular FFT
      7. Extraer features frecuenciales (dominante, bandas...)

    Args:
        raw_signal: Array 1D de muestras de aceleración en g o m/s².
                    Puede proceder de ADXL345, IEPE, otro MEMS o simulador.
        fs:         Frecuencia de muestreo en Hz.
        config:     Configuración opcional. Si None, usa valores por defecto.
        validate:   Si True, valida la señal antes de procesar.

    Returns:
        VibrationFeatures con features temporales, frecuenciales
        y propiedades de compatibilidad con el sistema legado.

    Raises:
        ValueError: Si la señal no es válida y validate=True.

    Ejemplo:
        >>> import numpy as np
        >>> from src.edge.signal_processing import process_vibration_signal
        >>>
        >>> fs = 3200.0
        >>> t  = np.linspace(0, 1, int(fs))
        >>> signal = 0.07 * np.sin(2 * np.pi * 50 * t)  # 50 Hz, 0.07g
        >>>
        >>> features = process_vibration_signal(signal, fs)
        >>> print(f"RMS: {features.RMS:.4f} g")
        >>> print(f"Kurtosis: {features.Kurtosis:.4f}")
        >>> print(f"Frecuencia dominante: {features.frequency.dominant_freq:.1f} Hz")
    """
    if config is None:
        config = SignalConfig(fs=fs)

    # 1. Validación de calidad
    if validate:
        quality = validate_signal(raw_signal, fs)
        if not quality["is_valid"]:
            raise ValueError(
                f"Señal no válida para procesamiento: "
                f"{'; '.join(quality['warnings'])}"
            )

    # 2. Detrend
    processed = detrend_signal(raw_signal, config.detrend_type)

    # 3. Filtro paso de banda
    # Ajustar límite superior si excede Nyquist
    nyq = fs / 2.0
    high = min(config.bandpass_high_hz, nyq * 0.95)
    processed = bandpass_filter(
        processed, fs,
        config.bandpass_low_hz, high,
        config.filter_order
    )

    # 4. Features temporales (sobre señal filtrada)
    time_feats = extract_time_features(processed)

    # 5 + 6 + 7. Features frecuenciales
    freq_feats = extract_frequency_features(processed, fs, config)

    return VibrationFeatures(
        time          = time_feats,
        frequency     = freq_feats,
        signal_length = len(raw_signal),
        fs            = fs,
        config        = config,
    )
