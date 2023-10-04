
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import detrend
from torch.utils.data import Dataset, DataLoader
import torch

class CustomDataLoader:
    def __init__(self, channels, epoch_transform, path):
        self.instances = None
        self.labels = None
        self.samples_by_channel = None # set by get_samples_by_channel
        self.samples = None
        self.channels = channels
        self.epoch_transform = None
        self.path = path

    # constructs an (N, d + 1) array where N is the number of samples and d is the number of channels. The last column is for the label
    def get_samples_by_channels(self)
        pass

    # constructs and returns an array of size (d, (N//s, s)) where d is nbr of channels, s is the samples_per_epoch and N is the number of samples in a channel  
    def defines_epochs(self, samples_per_epoch)
        pass

    # optional: do transformations on each epoch
    def transform_epochs(self, epoch):
        pass

    # constructs an array of size (N//s, a*s) where a is the number of desired epochs per instance. nbr_epochs is the number of epochs to use for one instance
    def get_data_instances(self, samples_per_epoch, nbr_epochs)
        pass

    @staticmethod
    def bandpass_filter(data, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs  # Nyquist frequency, which is half of fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y




# Dataset for shear building        

class ShearBuildingLoader(Dataset, CustomDataLoader):
    def __init__(self, epoch_transform):
        super().__init__(6, epoch_transform, "C:\\Users\\amroa\\Documents\\thesis\\sheartable")

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

    def defines_epochs(self, samples_per_epoch)
        if self.samples_by_channel is None:
            self.get_samples_by_channel()

        data = self.samples_by_channel[:, :-1]

        # define a wraparound in case the samples per epoch do not evenly divide the number of samples
        nbr_samples = data.shape[0]
        wraparound_amt = samples_per_epoch - (nbr_samples % samples_per_epoch)
        new_len = nbr_samples + wraparound_amt

        if wraparound_amt != 0:
            wraparound = self.samples_by_channels[0:wraparound_amt, :]
            data = np.vstack(( data, wraparound ))

        # now that the wraparound part is done, we reshape into the data into desired epochs 
        print(f"Epochs for {self.channels} channels")
        channel_epochs = []
        nbr_segs = int(new_len // samples_per_epoch) # could also be called nbr_epochs
        channels_epochs_sample =  [np.apply_along_axis(self.epoch_transform, axis = 1, arr=  data[i, :].reshape((nbr_segs, samples_per_epoch))  ) for i in range(self.channels)]

        # for the labels, we reshape into (nbr_segs or nbr_epochs, samples_)
        labels = (self.samples_by_channel[:, -1]).reshape((nbr_segs, samples_per_epoch))

        # we have to deal with heterogeneous rows by taking the majority element
        self.labels = np.apply_along_axis(lambda x: 1 if np.mean(x) >= 0.5 else 0, axis = 1, arr = labels)

        return np.array(channels_epochs_sample)

    # nbr_epochs is ignored for now (set to 5)
    def get_data_instances(self, samples_per_epoch, nbr_epochs):
        channels_epochs_sample = self.define_epochs(samples_per_epoch)
        epoch_sequences_all_channels = [] # double-list, i.e list of 6 lists, where each interior list stores several sequeneces of epochs 
        for i in range(self.channels):
            cur_epochs_sample = channels_epochs_sample[i] # get the data for the i'th channel
            num_epochs = cur_epochs_sample.shape[0]
            epoch_sequences = [] # stores sequences of epochs in a list
            
            # adapted from https://github.com/AmroAbdrabo/task4/blob/main/CNN.py
            for i in range(0, num_epochs):
                if i == 0:
                    epoch1, epoch2, epoch3, epoch4, epoch5 = 0, 0, 0, 1, 2
                elif i == 1:
                    epoch1, epoch2, epoch3, epoch4, epoch5 = 0, 0, 1, 2, 3
                elif i == num_epochs - 2:
                    epoch1, epoch2, epoch3, epoch4, epoch5 = i-2, i-1, i, i+1, i+1
                elif i == num_epochs - 1:
                    epoch1, epoch2, epoch3, epoch4, epoch5 = i-2, i-1, i, i, i
                else:
                    epoch1, epoch2, epoch3, epoch4, epoch5 = i-2, i-1, i, i+1, i+2
                
                epoch_sequence = np.concatenate((cur_epochs_sample[epoch1], cur_epochs_sample[epoch2], cur_epochs_sample[epoch3], cur_epochs_sample[epoch4], cur_epochs_sample[epoch5]))
                epoch_sequences.append(epoch_sequence)
            epoch_sequences_all_channels.append(np.array(epoch_sequences))
            
        # shape of dstack is (nbr_epochs, nbr_of_channels, epoch_shape[0]*5, epoch_shape[1]) note the *5 is due to the concatenation above
        self.instances =  np.stack(epoch_sequences_all_channels, axis = 1)

    def __len__(self):
        return self.instances.shape[0]

    def __getitem__(self, data):
        return self.instances[i], self.labels[i]