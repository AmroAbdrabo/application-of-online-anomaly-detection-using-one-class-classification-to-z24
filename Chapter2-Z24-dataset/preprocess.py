import numpy as np
from scipy.signal import butter, lfilter
from scipy.signal import detrend
import pandas as pd
import os
from scipy.io import loadmat

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

def preprocess_without_std(signal):
    low = 1 
    high  = 30
    sample_rate = 100
    return bandpass_filter(detrended_signal(signal), low, high, sample_rate)

"""
The following three methods from eda.ipynb in Chapter 2
"""

def data_preprocessing(data, trans_data, valid_columns = None, imputer = None, scaler = None): # for training data, default is None. For test, other parameters should be passed
    """
    Input:
      data: of size (544, 5, 49) where 544 is the number of observations (if training, 68 if test), 5 is the number of channels, and 49 is the features computed in utils4.py
      trans_data: (544, 30), where 30 is the length of the ratio of the PSD's of the reference signal R1V to the other channels
      valid_columns: None if training data. Otherwise, it is the columns that have less than 10% NAN values
      imputer: None if training data. Otherwise, it used to fill missing values using mean in the test data
      scaler: None if training data. Otherwise, it is used to scale test data  
    """
    import sklearn
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    
    X_flattened_unstd =   np.array([el.flatten() for el in data]) # unstd is short for unstandardized. Flatten all the channels into one 1D array for each observation
    X_flattened_unstd = np.hstack((X_flattened_unstd, trans_data))
    X_valid_cols = []
    X_imputed = []
    X_std = []

    ## Remove columns with too many NANs
    if valid_columns is None:
        # Create a boolean mask for NaN values
        nan_mask = np.isnan(X_flattened_unstd)

        # Check for NaN values in each column and print the column indices with NaN values
        columns_with_nan = np.any(nan_mask, axis=0)
        nan_columns_indices = np.where(columns_with_nan)[0]

        nan_per_column = [np.sum(np.isnan(X_flattened_unstd[:, el])) if el in nan_columns_indices else 0 for el in range(X_flattened_unstd.shape[1])]
        valid_columns = np.array(nan_per_column) < (0.1*(X_flattened_unstd.shape[0])) # if more than 10 percent of a column is NaN, we ignore it 
        print(np.sum(valid_columns)) # we will sadly let go of 5 columns

        X_valid_cols = X_flattened_unstd[:, valid_columns]
    else:
        X_valid_cols = X_flattened_unstd[:, valid_columns]

    ## Impute missing values
    if imputer is None:
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X_valid_cols)
    else:
        X_imputed = imputer.transform(X_valid_cols)

    ## Standardize the data
    if scaler is None:
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X_imputed)
    else:
        X_std = scaler.transform(X_imputed)

    return X_std, valid_columns, imputer, scaler

def get_dataframes(avt_files):
    """
    This method returns a list of dataframes # type: ignore
    A dataframe is of the form: # type: ignore
    R1V R2L ... R3V  (column names)
    0.1 0.3 ... 0.12
    0.4 0.1 ... 0.22 
    """
    good_sensors = ['R1V ', 'R2L ', 'R2T ', 'R2V ', 'R3V ']
    result = pd.DataFrame(columns=good_sensors)
    list_df = [] # should contain 17 entries, one for each damage state 

    # get pandas dataframes
    for idx, avt in enumerate(avt_files):
        # get index of the good sensor/positions
        arr_sensors = avt['labelshulp']
        bool_array = np.isin(arr_sensors, good_sensors) #  will contain True if entry corresponds to a good sensor
        indices_good_sensors = np.where(bool_array)[0]

        # Create a Pandas DataFrame
        df = pd.DataFrame(data=avt['data'][:, indices_good_sensors], columns=arr_sensors[indices_good_sensors])
        result = pd.concat([result, df], ignore_index=True)

        if (idx+1) % 9 == 0:
            # Most of the dataframes are 589824 except for 4 of them
            # Mirror the top part to the bottom to achive the same length
            top_rows = result.head(589824 - len(result))

            # Concatenate the sliced rows to the bottom of the original DataFrame
            new_res = pd.concat([result, top_rows], ignore_index=True)

            list_df.append(new_res)
            result = pd.DataFrame(columns=good_sensors)

    return list_df

def get_avt_files(root):
    folder_path = root
    file_list = os.listdir(folder_path)
    avt_files = []

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
                for avt_file in setup_filenames:
                    full_path_3 = os.path.join(full_path_2, avt_file) # full_path_3 is of the form C:\Users\amroa\Documents\thesis\data\01\avt\01setup09.mat
                    avt = loadmat(full_path_3)
                    avt_files.append(avt)
    return avt_files