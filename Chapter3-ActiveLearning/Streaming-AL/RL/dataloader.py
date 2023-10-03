
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import detrend

class DataLoader:
    def __init__(self, channels, path):
        self.instances = None
        self.samples_by_channel = None # set by get_samples_by_channel
        self.samples = None
        self.channels = channels
        self.path = path

    # constructs an (N, d + 1) array where N is the number of samples and d is the number of channels. The last column is for the label
    def get_samples_by_channels(self)
        pass

    # constructs and returns an array of size (d, (N//s, s)) where d is nbr of channels, s is the samples_per_epoch and N is the number of samples in a channel  
    def defines_epochs(self, samples_per_epoch)
        pass

    # constructs an array of size (N//s, a*s) where a is the number of desired epochs per instance 
    def get_data_instances(self)
        pass

    @staticmethod
    def bandpass_filter(data, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs  # Nyquist frequency, which is half of fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

class ShearBuildingLoader(DataLoader):
    def __init__(self):
        super().__init__(6, "C:\\Users\\amroa\\Documents\\thesis\\sheartable")

    def get_samples_by_channels(self):
        # check first if result was already computed
        if not (self.samples_by_channel is None):
            return self.samples_by_channel

        # folder path contains both damaged and undamged folders
        folder_path = self.path
        file_list = os.listdir(folder_path)
        dfs = [[],[]] # first list stores damaged dataframes, second list stores undamaged dataframes

        # walk inside data folder and convert each xlsx file to PD dataframe
        for idx, filename in enumerate(file_list):
            print(f"Appending {filename} first")
            full_path = os.path.join(folder_path, filename)
            file_list2 = os.listdir(full_path)
            for filename2 in file_list2:
                df = pd.read_excel(os.path.join(full_path, filename2), index_col=None, header=list(range(11))) 
                detrended_df = df.apply(lambda x: detrend(x), axis=0)   # bandpass filtering to 50 Hz completely removed the data so it will not be done
                filtered_df = detrended_df.apply(lambda x: bandpass_filter(x, 1, 100, 4096), axis=0) # introduces too many nans
                dfs[idx].append(filtered_df)
        
        # concatenate all the damaged cases together and then all the undamaged cases together
        damaged   = pd.concat(dfs[0], axis=0, ignore_index=True)
        undamaged = pd.concat(dfs[1], axis=0, ignore_index=True)

        # convert to numpy
        damaged_np = damaged.to_numpy()
        undamged_np = undamaged.to_numpy()

        # nbr of damage and healthy cases
        nbr_dam = damaged_np.shape[0]
        nbr_und = undamaged_np.shape[0]

        # labels
        labels_dam = np.ones(nbr_dam).reshape(nbr_dam, -1)
        labsls_und = np.zeros(nbr_und).reshape(nbr_und, -1)

        # concatenate labels to data horizontally
        data_dam    = np.hstack((damaged_np,   labels_dam))
        data_und  =   np.hstack((undamaged_np, labels_und))

        # save for later
        self.samples_by_channel = np.vstack((data_dam, data_und))

        return self.samples_by_channel
        

    def defines_epochs(self, samples_per_epoch)
        if self.samples_by_channel is None:
            get_samples_by_channel(self)

        data = self.samples_by_channel

        # define a wraparound in case the samples per epoch do not evenly divide the number of samples
        nbr_samples = len(data)
        wraparound_amt = nbr_samples % samples_per_epoch

        if wraparound_amt != 0:
            wraparound = self.samples_by_channels[0:wraparound_amt, :]
            data = np.vstack(( data, wraparound ))

        # now that the wraparound part is done






