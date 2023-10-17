from torch.utils.data import Dataset, DataLoader
import torch
from dataloader import Z24Loader
from cnn import transform_epoch
from cnn import CustomResNet
import numpy as np
from dataloader import ShearBuildingLoader

# Stores the features produced by CNN from cnn.py (after model has trained)
if __name__ == "__main__":
    building_type = 2 # set to 0 for shear loader, 1 for Z24, 2 for LUMO
    
    # size of each epoch (continuous segment/chunk of samples)
    z24_epoch_size = 16384
    shear_epoch_size = 16384 # should probably be smaller for shear since we have less data there
    lumo_epoch_size = int(16384 * (1651/100)) # ~270499

    z24_fs = 100 # sampling rate for z24
    shear_fs = 4096 #  .. and for shear building

    # Create the dataset and dataloader
    dataset_all = ShearBuildingLoader(shear_epoch_size, lambda epoch: transform_epoch(epoch, shear_fs)) if \
        building_type == 0 else Z24Loader(z24_epoch_size, lambda epoch: transform_epoch(epoch, z24_fs))
    dataset_all.get_data_instances(2, 1) 

    torch.cuda.empty_cache()
    dataloader = DataLoader(dataset_all, batch_size=4, shuffle=False)
    
    # Initialize the model and optimizer
    model = CustomResNet(version="50", num_classes=2).double()
    model.load_state_dict(torch.load('model_weights.pth'))

    print(f"CUDA availability {torch.cuda.is_available()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_features = []
    model.to(device)

    # Training loop
    model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            inputs, labels = batch

            inputs = inputs.to(device)
            
            outputs = model.features(inputs)
            # Detach, move to CPU, and convert to numpy
            outputs_np = outputs.cpu().numpy()
            all_features.append(outputs_np)
            
    all_features_array = np.concatenate(all_features, axis=0)
    np.save(f'features_{"building" if building_type == 0 else "bridge"}.npy', all_features_array)