""""
Reads sensor files. Root argument in fetch_signals assumes the following folder structure

- data
    - 01
        - avt
            - 01setup01
            - 01setup02
            ...
            - 01setup09
        - fvt
    - 02
        - avt
        - fvt
    - 03
        - avt
        - fvt
    ...
    - 17
        - avt
        - fvt

"""

import matplotlib.pyplot as plt
import os
import random
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks
from scipy.fftpack import fft, fftfreq
from scipy.signal import spectrogram
from scipy.io import loadmat

SIGLEN = 65536

# splits a signal, arr, into n equal parts (last part padded with zero in case of remainder)
def windowize(arr, n):
    remainder = len(arr) % n
    arr_padded = arr
    if remainder:
        padding_size = n - remainder
        arr_padded = np.concatenate((arr, np.zeros(padding_size)))

    # array after padding
    arr = arr_padded
    array_length = len(arr)
    subarray_size = int(array_length // n)
    subarrays = [arr[i:i + subarray_size] for i in range(0, array_length, subarray_size)]

    return subarrays

# on my computer root is 'C:\\Users\\amroa\\Documents\\thesis\\data'
# window size is 6000
def fetch_signals(root, window_size = 4096, train = True, train_split = 0.3):
    """
    window_size: length of a subdivision of original signal (6000 samples * (1/100) seconds/samples = 60 second sample)
    train: if True, then get training portion of data
    train_split: what proportion to use as training data (by default one third)
    """
    from scipy.io import loadmat
    signals = []
    folder_path = root
    file_list = os.listdir(folder_path)
    labels = []

    # for determinism
    random.seed(10)
    
    # walk inside data folder
    for filename in file_list:
        full_path = os.path.join(folder_path, filename)
        cur_label = int(filename)

        # walk inside data/[number]
        file_nest_list = os.listdir(full_path)
        for filename_1 in file_nest_list:
            if filename_1 != "avt":
                continue
            else:
                # full_path_2 is of the form C:\Users\amroa\Documents\thesis\data\01\avt
                full_path_2 = os.path.join(full_path, filename_1)
                setup_filenames = [filename for filename in os.listdir(full_path_2) if "setup" in filename]
                if train:
                    # take train_split percent of data
                    train_files = setup_filenames[0:int(len(setup_filenames)*train_split)]

                    # iterate through them
                    for file_train in train_files:
                        full_path_3 = os.path.join(full_path_2, file_train) # full_path_3 is of the form C:\Users\amroa\Documents\thesis\data\01\avt\01setup09.mat
                        
                        avt = loadmat(full_path_3)
                        avt_t = avt['data'].transpose()

                        # pad every signal to length 65536
                        padded_avt = [np.pad(arr, (0, SIGLEN - len(arr)), 'constant') for arr in avt_t]
                        avt_t = padded_avt

                        #split the signal into smaller signals of sample size window_size 
                        windows = [windowize(avt_t[i], int(len(avt_t[i])/window_size)) for i in range(len(avt_t))]
                        [[(signals.append(i), labels.append(cur_label)) for i in window] for window in windows] 
                else:
                    # take test_split percent of data
                    test_files = setup_filenames[int(len(setup_filenames)*train_split):]

                    # iterate through them
                    for file_test in test_files:
                        full_path_3 = os.path.join(full_path_2, file_test) # full_path_3 is of the form C:\Users\amroa\Documents\thesis\data\01\avt\01setup09.mat
                        
                        avt = loadmat(full_path_3)
                        avt_t = avt['data'].transpose()

                        # pad every signal to length 65536
                        padded_avt = [np.pad(arr, (0, SIGLEN - len(arr)), 'constant') for arr in avt_t]
                        avt_t = padded_avt

                        # split the signal into smaller signals of sample size window_size 
                        windows = [windowize(avt_t[i], int(len(avt_t[i])/window_size)) for i in range(len(avt_t))]
                        [[(signals.append(i), labels.append(cur_label)) for i in window] for window in windows] 
    return signals, labels

