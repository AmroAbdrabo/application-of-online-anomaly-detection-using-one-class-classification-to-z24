from torch.utils.data import Dataset, DataLoader
import torch
from dataloader import Z24Loader
from cnn import transform_epoch
from cnn import CustomResNet
import numpy as np
from dataloader import ShearBuildingLoader, LUMODataset
import os
from cnn import transform

# Change this if you want to produce weights from differently trained CNNs
weights_file = 'model_weights_R18_LUMO_2e.pth'
output_dir = "C:\\Users\\amroa\\Documents\\thesis\\LUMO_FEATURES"

# Stores the features produced by CNN from cnn.py (after model has trained)
if __name__ == "__main__":
    building_type = 1 # set to 0 for shear loader, 1 for Z24, 2 for LUMO
    
    # size of each epoch (continuous segment/chunk of samples)
    z24_epoch_size = 16384
    shear_epoch_size = 16384 # should probably be smaller for shear since we have less data there
    lumo_epoch_size = int(16384 * (412.75/100))

    z24_fs = 100 # sampling rate for z24
    shear_fs = 4096 #  .. and for shear building
    lumo_fs = 412.75

    # Create the dataset and dataloader
    dataset_all = ShearBuildingLoader(shear_epoch_size, lambda epoch: transform_epoch(epoch, shear_fs)) if \
        building_type == 0 else Z24Loader(z24_epoch_size, lambda epoch: transform_epoch(epoch, z24_fs)) if \
        building_type == 1 else LUMODataset(lumo_epoch_size, lambda epoch: transform_epoch(epoch, lumo_fs), transform) 
    dataset_all.get_data_instances(2, 1) 

    torch.cuda.empty_cache()
    dataloader = DataLoader(dataset_all, batch_size=4, shuffle=False)
    
    # Initialize the model and optimizer
    model = CustomResNet(version="18", num_classes=2)
    model.load_state_dict(torch.load(weights_file))

    print(f"CUDA availability {torch.cuda.is_available()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_features = []
    model.to(device)

    list_features = []
    list_labels = []
    # Training loop
    model.eval()
    with torch.inference_mode():
        for idx, batch in enumerate(dataloader):
            inputs, labels = batch
            inputs = inputs.float().to(device)

            outputs = model.features(inputs)
            # Detach, move to CPU, and convert to numpy
            outputs_np = outputs.cpu().numpy()

            # Loop through each instance and save as a numpy array
            for j, (output_instance, label) in enumerate(zip(outputs_np, labels)):
                #filename = os.path.join(output_dir, f"output_batch_{idx}_index_{j}_label_{int(label.item())}.npy")
                list_features.append(output_instance.squeeze())
                list_labels.append(int(label.item()))
                #np.save(filename, output_instance)
    np.save("resnet18_z24_feat.npy", np.hstack((np.vstack(list_features), np.array(list_labels).reshape(-1, 1))))