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
    
    # FT
    freqs = fftfreq(n, d=1/fs)
    magnitude = np.abs(fft(signal))
    
    # fundamental freq.
    fundamental_frequency = freqs[np.argmax(magnitude)]
    
    half_power = np.max(magnitude) / 2
    lower_bandwidth_freq = freqs[np.argmax(magnitude[magnitude > half_power])]
    upper_bandwidth_frequ = freqs[::-1][np.argmax(magnitude[::-1][magnitude > half_power])]
    bandwidth = upper_bandwidth_frequ - lower_bandwidth_freq
    
    # find peaks in magnitude spectrum, sort them in decreasing order of magnitude
    peak_indices, _ = find_peaks(magnitude)
    sorted_peaks = sorted(peak_indices, key=lambda i: -magnitude[i])
    harmonics = freqs[sorted_peaks]
    
    if len(harmonics) > 1:
        thd = np.sqrt(sum(magnitude[sorted_peaks[1:]]**2)) / magnitude[sorted_peaks[0]] # total harmonic distrotion
    else:
        thd = 0
    
    # spectral flatnes
    geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
    arithmetic_mean = np.mean(magnitude)
    spectral_flatness = geometric_mean / arithmetic_mean
    
    return [ 
        *magnitude,
        fundamental_frequency,
        bandwidth,
        thd,
        spectral_flatness
    ]

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
    autocorrelation_lag1 = np.correlate(signal, signal, mode='full')[len(signal)-1 + 1]

    return [
        root_mean_sq,
        peak_to_peak,
        crest_factor,
        form_factor,
        energy,
        power,
        zero_crossing_rate,
        autocorrelation_lag1
    ]