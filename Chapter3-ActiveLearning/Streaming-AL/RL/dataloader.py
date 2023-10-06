
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import detrend
from torch.utils.data import Dataset, DataLoader
import torch
import os

class CustomDataLoader:
    def __init__(self, channels, epoch_transform, path):
        self.instances = None
        self.labels = None
        self.samples_by_channel = None # set by get_samples_by_channel
        self.samples = None
        self.channels = channels
        self.epoch_transform = epoch_transform
        self.path = path

    # constructs an (N, d + 1) array where N is the number of samples and d is the number of channels. The last column is for the label
    def get_samples_by_channels(self):
        pass

    # constructs and returns an array of size (d, (N//s, s)) where d is nbr of channels, s is the samples_per_epoch and N is the number of samples in a channel  
    def define_epochs(self, samples_per_epoch):
        pass

    # optional: do transformations on each epoch
    def transform_epochs(self, epoch):
        pass

    # constructs an array of size (N//s, a*s) where a is the number of desired epochs per instance. nbr_epochs is the number of epochs to use for one instance
    def get_data_instances(self, samples_per_epoch, nbr_epochs):
        pass

    @staticmethod
    def bandpass_filter(data, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs  # Nyquist frequency, which is half of fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y
    
    @staticmethod # calculates indices of consecutive epochs
    def get_epoch_positions(i, nbr_epochs, tot_epochs):
        left_indices  = [max(i - el, 0) for el in range(1, int(nbr_epochs//2))]
        right_indices = [min(i + el, tot_epochs - 1) for el in range(int(nbr_epochs//2))] # includes i also
        return left_indices + right_indices

# Dataset for shear building        

class ShearBuildingLoader(Dataset, CustomDataLoader):
    def __init__(self, epoch_transform):
        super().__init__(6, epoch_transform, "C:\\Users\\amroa\\Documents\\thesis\\sheartable")

    def get_samples_by_channels(self):
        try:
            shear_build_samp_by_chnl = np.load("shear_build_samp_by_chnl.npy")
            self.samples_by_channel = shear_build_samp_by_chnl
        except:
            print("Could not read pickled shear_build_samp_by_chnl.npy")
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
                #filtered_df = detrended_df.apply(lambda x: CustomDataLoader.bandpass_filter(x, 1, 100, 4096), axis=0) # introduces too many nans
                filtered_df = detrended_df
                filtered_df.columns = range(7) # ignore column names to avoid problems later
                filtered_df = filtered_df.drop(columns=filtered_df.columns[0]) # make sure to drop time as it is not needed
                dfs[idx].append(filtered_df)
        
        # concatenate all the damaged cases together and then all the undamaged cases together
        damaged   = pd.concat(dfs[0], axis=0, ignore_index=True)
        undamaged = pd.concat(dfs[1], axis=0, ignore_index=True)

        # convert to numpy
        damaged_np = damaged.astype(np.float64).to_numpy()
        undamaged_np = undamaged.astype(np.float64).to_numpy()

        print(damaged_np.shape)
        print(undamaged_np.shape)

        # nbr of damage and healthy cases
        nbr_dam = damaged_np.shape[0]
        nbr_und = undamaged_np.shape[0]

        # labels
        labels_dam = np.ones(nbr_dam).reshape(nbr_dam, -1)
        labels_und = np.zeros(nbr_und).reshape(nbr_und, -1)

        # concatenate labels to data horizontally
        data_dam    = np.hstack((damaged_np,   labels_dam))
        data_und  =   np.hstack((undamaged_np, labels_und))
        print(data_dam.shape)
        print(data_und.shape)
        self.samples_by_channel = np.vstack((data_dam, data_und))
        # save for later
        np.save("shear_build_samp_by_chnl.npy", self.samples_by_channel)
    def define_epochs(self, samples_per_epoch):
        if self.samples_by_channel is None:
            self.get_samples_by_channels()

        data = self.samples_by_channel[:, :-1]

        # define a wraparound in case the samples per epoch do not evenly divide the number of samples
        nbr_samples = data.shape[0]
        wraparound_amt = samples_per_epoch - (nbr_samples % samples_per_epoch)
        new_len = nbr_samples + wraparound_amt

        if wraparound_amt != 0:
            wraparound = self.samples_by_channel[0:wraparound_amt, :]
            self.samples_by_channel = np.vstack(( self.samples_by_channel, wraparound ))
            data = self.samples_by_channel[:, :-1]

        # now that the wraparound part is done, we reshape into the data into desired epochs 
        print(f"Epochs for {self.channels} channels")
        
        nbr_segs = int(new_len // samples_per_epoch) # could also be called nbr_epochs

        # list of length number of channel, where each element is of size (nbr_segs, samples_per_epoch) if transform is identity otherwise each element is of size (nbr_segs, shape of return value of transform on one epoch)
        channels_epochs_sample =  [np.apply_along_axis(self.epoch_transform, axis = 1, arr=  data[:, i].reshape((nbr_segs, samples_per_epoch))  ) for i in range(self.channels)]

        # for the labels, we reshape into (nbr_segs or nbr_epochs, samples_)
        labels = (self.samples_by_channel[:, -1]).reshape((nbr_segs, samples_per_epoch))

        # we have to deal with heterogeneous rows by taking the majority element
        self.labels = np.apply_along_axis(lambda x: 1 if np.mean(x) >= 0.5 else 0, axis = 1, arr = labels).astype(np.int64)

        return np.array(channels_epochs_sample) # shape: (channels, nbr_epochs, samples or shape of return value of epoch_transform)

    # nbr_epochs is ignored for now (set to 5). nbr_epochs should be odd
    def get_data_instances(self, train, samples_per_epoch, nbr_epochs):
        channels_epochs_sample = self.define_epochs(samples_per_epoch) # (d, N//s, s or (3d array in case epoch transform ret RGB image))  
        num_epochs = channels_epochs_sample.shape[1] 
        epoch_sequences = [] # stores sequences of epochs in a list
        
        # adapted from https://github.com/AmroAbdrabo/task4/blob/main/CNN.py
        for i in range(0, num_epochs):
            conseutive_epoch_indices = CustomDataLoader.get_epoch_positions(i, nbr_epochs, tot_epochs = num_epochs) # nbr_epochs is the number of consecutive epochs to consider for each instance
            arr = []
            for j in range(self.channels):
                cur_epochs_sample = channels_epochs_sample[j] # get the data for the i'th channel
                arr.append(np.concatenate([cur_epochs_sample[ep] for ep in conseutive_epoch_indices])) # shape epoch_shape[0]*5, epoch_shape[1], epoch_shape[2]
            epochs_all_channel_center_i = np.concatenate(arr) # shape epoch_shape[0]*5*channels, epoch_shape[1], epoch_shape[2], note epoch_shape[2] is RGB information in case we use pcolormesh
            epoch_sequences.append(epochs_all_channel_center_i.transpose((2, 0, 1))) # RGB channel first since pytorch requires it that way 
        
        # shape of instances is (nbr_epochs, epoch_shapa[2], epoch_shape[0]*5*nbr_channels, epoch_shape[1]) note the *5 is due to the concatenation above and that epoch_shape[2] is most likely 3 for the number of channels in an RGB image
        np.random.seed(42)
        self.instances =  np.array(epoch_sequences).astype(np.float64)
        nbr_inst = self.instances.shape[0]
        num_elements = int(0.7 * nbr_inst) # for training data we use 70%
        if train:
            selected_indices = np.random.choice(nbr_inst, size=num_elements, replace=False)
            subset_train = self.instances[selected_indices]
            self.instances = subset_train
        else:
            selected_indices = np.random.choice(nbr_inst, size=num_elements, replace=False)
            # get the numbers not in selected_indices
            not_selected = np.setdiff1d(np.arange(nbr_inst), selected_indices)
            subset_test = self.instances[not_selected]
            self.instances = subset_test

    def __len__(self):
        return self.instances.shape[0]

    def __getitem__(self, idx):
        return self.instances[idx], self.labels[idx]