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


if __name__ == "__main__"
    # function to transform an epoch of acceelration
    def transform_accel_epoch(epoch):
        # calculate the spectrogram
        f, t_spec, Sxx = spectrogram(y, 4096)
        
        # select subplot
        plt.subplot(n_rows, n_cols, i+1)
        plt.pcolormesh(t_spec, f, 10*np.log10(Sxx), shading = 'gouraud')

    # Create the dataset and dataloader
    dataset = ShearBuildingLoader(data, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)