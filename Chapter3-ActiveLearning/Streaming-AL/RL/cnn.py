from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision.models as models
from dataloader import ShearBuildingLoader

class CustomResNet(nn.Module):
    def __init__(self, version="18", num_classes=1000):
        super(CustomResNet, self).__init__()
        
        # Define the first conv layer
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Load pre-trained ResNet (for other layers)
        if version == "18":
            resnet = models.resnet18(pretrained=False)
        else:
            resnet = models.resnet50(pretrained=False)
            
        # Replace the first convolutional layer in the pre-trained model with our 6-channel one
        resnet.conv1 = self.conv1
        
        # Use all layers of the ResNet model, but don't include the fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Create a new fully connected layer
        self.fc = nn.Linear(512 * (4 if version == "50" else 1), num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# convert to numpy array for usage in the CNN
def pcolormesh_to_array(quadmesh):
    """
    Convert the image plotted by pcolormesh on an Axes instance to a 2D numpy array.

    Parameters:
    - ax: matplotlib.axes.Axes, the Axes instance containing the pcolormesh plot.

    Returns:
    - img_array: 2D numpy array, the image array.
    """

    # Get the data from the QuadMesh
    data_array = quadmesh.get_array().data

    # Get the number of rows and columns
    nrows = quadmesh._meshHeight  
    ncols = quadmesh._meshWidth 
    

    # Convert the 1D array back to the original 2D shape
    data_2d = data_array.reshape(nrows, ncols)

    # Get the colormap and normalization from the QuadMesh
    cmap = quadmesh.get_cmap()
    norm = quadmesh.norm

    # Convert the data to RGBA values
    rgba_data = cmap(norm(data_2d))

    # Create an image array by getting the RGB values
    img_array = rgba_data[:, :, :3]

    return img_array

if __name__ == "__main__"
    # function to transform an epoch of acceelration of the shearr buidling
    def transform_epoch_shearbuilding(epoch):
        # calculate the spectrogram
        fs = 4096
        f, t_spec, Sxx = spectrogram(y, fs)
        
        # select subplot
        plt.pcolormesh(t_spec, f, 10*np.log10(Sxx), shading = 'gouraud')

    # Create the dataset and dataloader
    dataset = ShearBuildingLoader(data, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)