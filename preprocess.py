import numpy as np
from scipy.signal import butter, lfilter
from scipy.signal import detrend


def standardize_signal(signal):
    mean_signal = np.mean(signal)
    std_signal = np.std(signal)
    standard_signal = (signal - mean_signal) / std_signal
    
    return standard_signal

def remove_dc_component(signal):
    dc_removed_signal = signal - np.mean(signal)
    return dc_removed_signal


def detrended_signal(filtered_signal):
    return detrend(filtered_signal)

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Design and apply a Butterworth bandpass filter.
    Parameters:
        data (array): The signal to be filtered.
        lowcut (float): The lower frequency cut-off.
        highcut (float): The higher frequency cut-off.
        fs (int): The sampling rate of the data.
        order (int, optional): Order of the Butterworth filter. Default is 4.
    Returns:
        array: Filtered data.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_data = lfilter(b, a, data)
    return filtered_data

def preprocess(signal):
    low = 1 
    high  = 30
    sample_rate = 100
    return standardize_signal(detrended_signal(bandpass_filter(remove_dc_component(signal), low, high, sample_rate)))