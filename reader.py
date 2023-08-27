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

# splits a signal, arr, into n equal parts (except possibly for the last part due to remainder)
def windowize(arr, n):
    array_length = len(arr)
    subarray_size = int(array_length // n)
    remainder = array_length % n
    subarrays = [arr[i:i + subarray_size] for i in range(0, array_length, subarray_size)]
    if remainder:
        subarrays[-2] = np.concatenate((subarrays[-2], subarrays[-1]))
        subarrays.pop()
    return subarrays

# on my computer root is 'C:\\Users\\amroa\\Documents\\thesis\\data'
# window size is 6000
def fetch_signals(root, window_size = 6000, subset = True):
    """
    subset: if True, then get training portion of data
    """
    from scipy.io import loadmat
    signals = []
    folder_path = root
    file_list = os.listdir(folder_path)

    # for determinism
    random.seed(10)

    # walk inside data folder
    for filename in file_list:
        full_path = os.path.join(folder_path, filename)

        # walk inside data/[number]
        file_nest_list = os.listdir(full_path)
        for filename_1 in file_nest_list:
            if filename_1 != "avt":
                continue
            else:
                # full_path_2 is of the form C:\Users\amroa\Documents\thesis\data\01\avt
                full_path_2 = os.path.join(full_path, filename_1)
                setup_filenames = [filename for filename in os.listdir(full_path_2) if "setup" in filename]
                if subset:
                    random_setup_filename = random.choice(setup_filenames)
                    full_path_3 = os.path.join(full_path_2, random_setup_filename)
                    # full_path_3 is of the form C:\Users\amroa\Documents\thesis\data\01\avt\01setup09.mat

                    avt = loadmat(full_path_3)
                    avt_t = avt['data'].transpose()

                    # split the signal into smaller signals of sample size window_size 
                    windows = windowize(avt_t[0], len(avt_t[0])/window_size)
                    [signals.append(i) for i in windows] 
                else:
                    for filename_2 in setup_filenames:
                        full_path_3 = os.path.join(full_path_2, random_setup_filename)
                        # full_path_3 is of the form C:\Users\amroa\Documents\thesis\data\01\avt\01setup09.mat
                        avt = loadmat(full_path_3)
                        avt_t = avt['data'].transpose()
                        
                        # split the signal into smaller signals of sample size window_size 
                        windows = windowize(avt_t[1], len(avt_t[1])/window_size)  # this can be done for 1 to len(avt_t) for complete set of test cases
                        [signals.append(i) for i in windows] 
    return signals
