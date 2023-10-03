import numpy as np
from scipy.signal import find_peaks
from scipy.fftpack import fft, fftfreq

def frequency_domain_characteristics(signal, fs=100):
    """
    frequency domain characteristics.
    :param signal: List or numpy array representing the signal.
    :param fs: Sampling frequency of the signal.
    :return: List containing frequency domain characteristics.
    """
    n = len(signal)
    frequencies = np.fft.fftfreq(n, 1/fs)
    yf = fft(signal)

    # fundamental freq.
    fundamental_frequency = frequencies[np.argmax(np.abs(yf))]

    # Amplitude Spectrum
    amplitude_spectrum = np.abs(yf)[:n//2]

    # Phase Spectrum
    phase_spectrum = np.angle(yf)[:n//2]

    # Power Spectrum
    power_spectrum = amplitude_spectrum ** 2

    # Spectral Centroid
    spectral_centroid = np.sum(frequencies[:n//2] * amplitude_spectrum) / np.sum(amplitude_spectrum)

    # Spectral Spread
    spectral_spread = np.sqrt(np.sum((frequencies[:n//2] - spectral_centroid)**2 * amplitude_spectrum) / np.sum(amplitude_spectrum))

    # Feature Vector
    feature_vector = np.concatenate([fundamental_frequency, amplitude_spectrum, phase_spectrum, [power_spectrum.mean(), spectral_centroid, spectral_spread]])

    return feature_vector

def time_domain_characteristics(signal):
    # Convert the signal to a numpy array for computation
    signal = np.array(signal)
    root_mean_sq = np.sqrt(np.mean(np.square(signal))) 
    peak_to_peak = np.ptp(signal)  # ptp stands for 'peak to peak'
    crest_factor = np.max(np.abs(signal)) / root_mean_sq
    mean_value = np.mean(signal)
    form_factor = root_mean_sq / mean_value if mean_value != 0 else np.inf
    energy = np.sum(np.square(signal))
    power = energy / len(signal)
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    zero_crossing_rate = len(zero_crossings) / len(signal)
    # autocorrelation (for lag = 1 as example)
    autocorrelation_lag1 = np.sum(signal[:-1] * signal[1:])

    return np.array([
        root_mean_sq,
        peak_to_peak,
        crest_factor,
        form_factor,
        energy,
        power,
        zero_crossing_rate,
        autocorrelation_lag1
    ])