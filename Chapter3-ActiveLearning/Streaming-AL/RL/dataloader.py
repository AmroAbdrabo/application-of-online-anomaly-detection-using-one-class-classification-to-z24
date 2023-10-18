
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import detrend
from torch.utils.data import Dataset, DataLoader
import torch
import sys
import os
import gc
from PIL import Image

class CustomDataLoader:
    def __init__(self, channels, epoch_size, epoch_transform, path):
        self.instances = None
        self.labels = None
        self.samples_by_channel = None # set by get_samples_by_channel
        self.samples = None
        self.channels = channels
        self.epoch_size = epoch_size # number of samples in each epoch
        self.epoch_transform = epoch_transform
        self.path = path
        self.message = "Custom dataloader"
        self.split = None # amount of training data

    # constructs an (N, d + 1) array where N is the number of samples and d is the number of channels. The last column is for the label
    def get_samples_by_channels(self):
        pass

    # constructs and returns an array of size (d, (N//s, s)) where d is nbr of channels, s is the samples_per_epoch and N is the number of samples in a channel  
    def define_epochs(self, samples_per_epoch):
        pass

    # constructs an array of size (N//s, a*s) where a is the number of desired epochs per instance. nbr_epochs is the number of epochs to use for one instance
    # train_test_all is 0 if train, 1 if test, and 2 if all
    def get_data_instances(self, train_test_all, nbr_epochs):
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
        right_indices = [min(i + el, tot_epochs - 1) for el in range(1, int(nbr_epochs//2))] # includes i also
        return left_indices + [i] + right_indices
    
    def define_epochs(self):

        # same as shear building
        samples_per_epoch = self.epoch_size
        if self.samples_by_channel is None:
            self.get_samples_by_channels()

        # get the samples by channel, but exclude labels
        data = self.samples_by_channel[:, :-1]

        # define a wraparound in case the samples per epoch do not evenly divide the number of samples
        nbr_samples = data.shape[0]
        print(nbr_samples)
        print(samples_per_epoch)
        wraparound_amt = samples_per_epoch - (nbr_samples % samples_per_epoch) 
        new_len = nbr_samples + wraparound_amt if (nbr_samples % samples_per_epoch) != 0 else nbr_samples

        # technically no wraparound should be generated for Z24 as the method get_dataframes ensures that all lengths are powers of 2
        if (nbr_samples % samples_per_epoch) != 0:
            print("Wrap-around occurred")
            print(type(wraparound_amt))
            print(wraparound_amt)
            wraparound = self.samples_by_channel[0:wraparound_amt, :]
            self.samples_by_channel = np.vstack(( self.samples_by_channel, wraparound ))
            data = self.samples_by_channel[:, :-1]

        # now that the wraparound part is done, we reshape into the data into desired epochs 
        print(f"Epochs for {self.channels} {self.message}")
        
        nbr_segs = int(new_len // samples_per_epoch) # could also be called nbr_epochs

        # list of length number of channel, where each element is of size (nbr_segs, samples_per_epoch) if transform is identity otherwise each element is of size (nbr_segs, shape of return value of transform on one epoch)
        channels_epochs_sample =  [np.apply_along_axis(self.epoch_transform, axis = 1, arr=  data[:, i].reshape((nbr_segs, samples_per_epoch))  ) for i in range(self.channels)]

        # for the labels, we reshape into (nbr_segs or nbr_epochs, samples_)
        labels = (self.samples_by_channel[:, -1]).reshape((nbr_segs, samples_per_epoch))

        # we have to deal with heterogeneous rows by taking the majority element
        self.labels = np.apply_along_axis(lambda x: 1 if np.mean(x) >= 0.5 else 0, axis = 1, arr = labels).astype(np.int64)

        return np.array(channels_epochs_sample) # shape: (channels, nbr_epochs, samples_per_epoch or shape of return value of epoch_transform)
    
    # DONE: place it in the super class CustomDataLoader
    def get_data_instances(self, train_test_all, nbr_epochs):
        
        channels_epochs_sample = self.define_epochs() # (d, N//s, s or (3d array in case epoch transform ret RGB image))  
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
        num_elements = int(self.split * nbr_inst) # for training data we use 90%
        selected_indices = np.random.choice(nbr_inst, size=num_elements, replace=False)
        if train_test_all == 0:
            subset_train = self.instances[selected_indices]
            self.labels = self.labels[selected_indices]
            self.instances = subset_train
        elif train_test_all == 1:
            # get the numbers not in selected_indices
            not_selected = np.setdiff1d(np.arange(nbr_inst), selected_indices)
            subset_test = self.instances[not_selected]
            self.labels = self.labels[not_selected]
            self.instances = subset_test


# Dataset for shear building        

class ShearBuildingLoader(Dataset, CustomDataLoader):
    def __init__(self, epoch_size, epoch_transform):
        CustomDataLoader.__init__(self, 6, epoch_size, epoch_transform, "C:\\Users\\amroa\\Documents\\thesis\\sheartable")
        self.message = "Shearloader"
        self.split = 0.7

    def get_samples_by_channels(self):
        try:
            shear_build_samp_by_chnl = np.load("shear_build_samp_by_chnl.npy")
            self.samples_by_channel = shear_build_samp_by_chnl
            return 
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

    def __len__(self):
        return self.instances.shape[0]

    def __getitem__(self, idx):
        return self.instances[idx], self.labels[idx]

from scipy.io import loadmat

Z24_HEALTHY_STATES = np.arange(8)

class Z24Loader(CustomDataLoader, Dataset):
    def __init__(self, epoch_size, epoch_transform):
        CustomDataLoader.__init__(self, 5, epoch_size, epoch_transform, "C:\\Users\\amroa\\Documents\\thesis\\data")
        sys.path.append('.\\Chapter2-Z24-dataset')
        self.message = "Z24"
        self.split = 0.9 # percent of training data

    def get_avt_files(self, root):
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

    def get_dataframes(self, avt_files):
        """
        This method returns a list of dataframes # type: ignore
        A dataframe is of the form: # type: ignore
        R1V R2L ... R3V  (column names)
        0.1 0.3 ... 0.12
        0.4 0.1 ... 0.22 
        """
        from preprocess import preprocess, preprocess_without_std # type: ignore
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
                new_res = new_res.apply(lambda x: preprocess_without_std(x), axis=0)   # bandpass filtering to 50 Hz completely removed the data so it will not be done

                list_df.append(new_res)
                result = pd.DataFrame(columns=good_sensors)

        return list_df

    @staticmethod # generate binary labels out of damage scenario labels
    def binarize(label):
        if label in Z24_HEALTHY_STATES: 
            return 0
        else:
            return 1
    def get_samples_by_channels(self):
        try:
            z24_samp_by_chnl = np.load("z24_samp_by_chnl.npy")
            self.samples_by_channel = z24_samp_by_chnl
            return 
        except:
            print("Could not read pickled z24_samp_by_chnl.npy")

        # read the avt files in MAT format
        avt_files = self.get_avt_files("C:\\Users\\amroa\\Documents\\thesis\\data")

        # get 17 pandas dataframes (one for each damage scenario) 
        dfs = self.get_dataframes(avt_files)

        # get the length of each damage scenario (nbr of rows of each dataframe)
        label_lengths = [len(df) for df in dfs]

        # create a list of labels e.g. [0,0,0,...1,1,1,1,1....2,2,2,......16,16,16] --> binarize labels --> [0,0,0,0, ...., 0,0,0, ... 1,1,1,1,1,1]
        labels = np.concatenate([np.full(label_lengths[i], Z24Loader.binarize(i)) for i in range(len(dfs))])

        # total number of acceleration samples (each sampling point has a label)
        nbr_labels = len(labels) 

        # concatenate all dataframes vertically into a large dataframe
        df_samples_by_channel  = pd.concat(dfs, axis=0, ignore_index=True)

        # convert it to numpy
        samples_by_channels = df_samples_by_channel.astype(np.float64).to_numpy()

        # append the labels to the numpy array
        self.samples_by_channel = np.hstack((samples_by_channels, labels.reshape(nbr_labels, -1)))
        
        # save for faster loading in the try block at the top of this method 
        np.save("z24_samp_by_chnl.npy", self.samples_by_channel)

    # Same as ShearBuildingLoader get_instances 
    
    def __len__(self):
        return self.instances.shape[0]

    def __getitem__(self, idx):
        return self.instances[idx], self.labels[idx]
    

# Based on the data provided by Yves
class BuildingLoader(Dataset, CustomDataLoader):
    def __init__(self, epoch_size, epoch_transform):
        CustomDataLoader.__init__(self, 3, epoch_size, epoch_transform, "C:\\Users\\amroa\\Documents\\thesis\\ASCE_benchmark.json")
        self.message = "Building dataloader"
        self.split = 0.7 # percent of training data

    def get_samples_by_channels(self):
        try:
            build_samp_by_chnl = np.load("yves_samp_by_chnl.npy")
            self.samples_by_channel = build_samp_by_chnl
            return 
        except:
            print("Could not read pickled yves_samp_by_chnl.npy")

        # check first if result was already computed
        if not (self.samples_by_channel is None):
            return

        df = pd.read_json(self.path)
        arrs = []
        arrs_labels = []
        for label in ['Healty', 'Damage1', 'Damage2']:
            samples_per_channel_i = np.array([df[label][i] for i in range(3)]).transpose()
            labels_i = np.full(samples_per_channel_i.shape[0], 0 if label == 'Healty' else 1) 
            arrs.append(samples_per_channel_i)
            arrs_labels.append(labels_i)

        samples_per_channel = np.concatenate(arrs, axis = 0)
        all_labels = np.concatenate(arrs_labels, axis = 0)
        self.samples_by_channel = np.hstack([samples_per_channel, all_labels.reshape(-1, 1)])

        # save for later
        np.save("yves_samp_by_chnl.npy", self.samples_by_channel)

    def __len__(self):
        return self.instances.shape[0]

    def __getitem__(self, idx):
        return self.instances[idx], self.labels[idx]
    
# since the LUMO labels are large we keep them once in the program
lumo_labels = np.load("labels_lumo.npy").astype(np.float64)

class LUMODataset(CustomDataLoader, Dataset):
    def __init__(self, epoch_size, epoch_transform, tensor_transform):
        CustomDataLoader.__init__(self, 11, epoch_size, epoch_transform, None) #  mentions 11 but only 6 used for storage reasons
        self.message = "LUMO"
        self.split = 0.7 # percent of training data
        self.file_to_state = dict() # mapping of measurement file to state of the tower

        # We use measurements from October 2020 only since all data would be too much and because October contains healthy and damaged
        self.meta_file = "C:\\Users\\amroa\\Documents\\thesis\\LUMO\\SHMTS_202010_meta_struct.mat"
        
        # where the actual measurements are stored
        self.structural_data_root = "D:\\LUMO\\10\\2020\\10" # 10 for October

        # given the size of this dataset, we only use specific days (each day is 144 files so we are sparing with the days)
        self.measured_dates = [LUMODataset.date_generate(2020, 10, i) for i in [1, 2, 3, 14, 15, 16]]

        # read the state of the files (state of the building being measured in the files)
        self.read_file_to_state()

        # image directory
        self.img_dir = "C:\\Users\\amroa\\Documents\\thesis\\IMGC"
        self.len = None
        self.selected_indices = None
        self.transform = tensor_transform
        self.mid = int(epoch_size//2)
        
        print("Done")

    @staticmethod
    def convert_ascii_to_str(ascii_list):
    # Convert a list of ASCII values to a string
        return ''.join(chr(int(val)) for val in ascii_list) 

    def read_file_to_state(self):
        import h5py
        with h5py.File(self.meta_file, 'r') as file:
            # Checking if 'Dat' is a key in the file
            if 'Dat' in file:
                dat_group = file['Dat']
            
                # Printing keys within the 'Dat' group
                for key in dat_group.keys():
                    # If the key is "Info", and it's a Dataset, then extract its content
                    item = dat_group[key]
                    if isinstance(item, h5py.Dataset) and key == "Info":
                        #print(f"Dataset {key} shape: {item.shape}"
                        # Iterate over each measurement
                        for col_idx in range(item.shape[1]):
                            folder_ref, state_ref = item[:, col_idx]
                            folder_name_ascii = file[folder_ref][()]
                            folder_name = LUMODataset.convert_ascii_to_str(folder_name_ascii.flatten())
                            
                            state_number = file[state_ref][()]
                            
                            #print(f"Measurement {col_idx+1}: Folder Name = {folder_name}, State Number = {state_number[0]}")
                            if any([folder_name.startswith(x) for x in self.measured_dates]): # only keep allowed dates
                                self.file_to_state[folder_name] = int(state_number[0][0])

    @staticmethod
    def date_generate(year, month, day):
        """
        Generate a string of the format SHMTS_YYYYMMDDHHMM
        using the provided year, month, day, hour, and minute.
        """
        return f"SHMTS_{year:04}{month:02}{day:02}"

    def get_samples_by_channels_file(self, file):
        print(f"Getting samples of file {file}")
        import h5py
        from scipy.signal import detrend
        with h5py.File(file, 'r') as file: 
            dat_group = file['Dat']['Data']
            x = dat_group[:]
            # detrend then bandpass filter
            x = x[::2, ::4] # keep only 11 channels and reduce sampling rate to 1651/4 = 412.75
            x = np.apply_along_axis(lambda r: CustomDataLoader.bandpass_filter(detrend(r), 0.5, 120, 412.75, order = 4), 1, x)
            gc.collect()  # Call garbage collection after processing the file
            # x.shape is (22, 990600) so we must transpose it 
            return x.transpose()

    def get_samples_by_channels(self):
        # check if it was already computed
        try:
            all_days = [np.load(f"lumo_{date}_samp_by_chnl.npy") for date in self.measured_dates]
            self.samples_by_channel = np.vstack(all_days) # most likely will not even get to this part
            print(self.samples_by_channel.shape)
            return 
        except Exception as e:
            print(f"Could not read pickled lumo_samp_by_chnl.npy. Error {e}")

        checkpoint = 0 # checkpoint is assuming a file fetch fails. Assume at iter=2 a fetch fails. Then set this to 2 
        iter = -1
        for date in self.measured_dates:
            samples_by_chnl_all = [] # for a single day 
            labels_all = [] #  likewise just for a day
            print(f"Processing day {date}")
            keys = [key for key in self.file_to_state if key.startswith(date)]
            for filename in keys: # each key is a filename
                # update current loop count
                iter = iter + 1
                state = self.file_to_state[filename]

                if (state < 2) or (iter < checkpoint):
                    # if data is not classified or is corrupted
                    continue
                
                # get samples_by_channels for the particular file 
                file_structural = os.path.join(self.structural_data_root, filename) + ".mat"

                try:
                    samples_by_channel_file = self.get_samples_by_channels_file(file_structural)
                except:
                    print("Error occurred during .mat file fetch. Checkpointing progress up to but not including iteration {iter}...")
                    checkpoint = np.hstack([np.vstack(samples_by_chnl_all), np.concatenate(labels_all).reshape(-1, 1)])
                    np.save(f"lumo_{iter}_samp_by_chnl.npy", checkpoint)

                samples_by_chnl_all.append(samples_by_channel_file)
                labels_all.append(np.full(samples_by_channel_file.shape[0], 0 if state==2 else 1)) # 2 refers to healthy state see readme file in https://data.uni-hannover.de/dataset/lumo/resource/bd0a6d0a-3ff3-4780-91cc-1d816ab39fb9
                
                gc.collect()  # Call garbage collection at the end of each iteration

            samples_by_channel = np.hstack([np.vstack(samples_by_chnl_all), np.concatenate(labels_all).reshape(-1, 1)])
            np.save(f"lumo_{date}_samp_by_chnl.npy", samples_by_channel)
            gc.collect()  # Call garbage collection at the end of the function
        
        # now if everything works
        all_days = [np.load(f"lumo_{date}_samp_by_chnl.npy") for date in self.measured_dates]
        self.samples_by_channel = np.vstack(all_days) # most likely will not even get to this part

    # Less memory intensive version, which saves images
    def define_epochs(self):
        if self.samples_by_channel is None:
            self.get_samples_by_channels()

        samples_per_epoch = self.epoch_size
        nbr_samples = self.samples_by_channel.shape[0]
        wraparound_amt = samples_per_epoch - (nbr_samples % samples_per_epoch) 

        if (nbr_samples % samples_per_epoch) != 0:
            print("Wrap-around occurred")
            wraparound = self.samples_by_channel[:wraparound_amt, :]
            self.samples_by_channel = np.vstack((self.samples_by_channel, wraparound))
        else:
            wraparound_amt = 0

        new_len = nbr_samples + wraparound_amt
        nbr_segs = new_len // samples_per_epoch

        print(f"Epochs for {self.channels} {self.message}")
        
        # classical for loop because memory is becoming problematic
        #for i in range(self.channels):
        for i in range(0, 6): # otherwise it's just too many channels
            row_nbr = 0
            for row in self.samples_by_channel[:, i].reshape((nbr_segs, samples_per_epoch)):
                img = self.epoch_transform(row)
                #np.save(f"D:\\LUMO\\IMG\\img_ch_{i}_row_{row_nbr}.npy", img)
                image_data = (img * 255).astype(np.uint8)
                image = Image.fromarray(image_data, 'RGB')  # Specify 'RGB' mode
                image.save(f"D:\\LUMO\\IMG\\img_ch_{i}_row_{row_nbr}.png")
                row_nbr = row_nbr + 1
            gc.collect()

        print("Saving images done")

    def get_data_instances(self, train_test_all, nbr_epochs):
        # IMPORTANT: self.define needs to be called separately by the caller (cnn.py) and only ONCE before this method
        tot_len = len(os.listdir(self.img_dir))
        np.random.seed(42)
        training_indices = np.random.choice(tot_len, size=int(self.split*tot_len), replace=False)
        test_indices = np.setdiff1d(np.arange(tot_len), training_indices)

        if train_test_all == 0:
            self.len = int(tot_len*self.split)
            self.selected_indices = np.sort(training_indices)
        elif train_test_all == 1:
            self.len = int(tot_len*(1- self.split))
            self.selected_indices = np.sort(test_indices)
        else:
            self.len = int(tot_len)
            self.selected_indices = np.arange(tot_len)
        return

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        row = self.selected_indices[idx]
        file = f"img_row_{row}.png"
        path = os.path.join(self.img_dir,file)
        label = lumo_labels[row*self.epoch_size:(row+1)*self.epoch_size][self.mid]
        return self.transform(Image.open(path)), label