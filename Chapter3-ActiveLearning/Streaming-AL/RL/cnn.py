from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from dataloader import ShearBuildingLoader
import matplotlib.pyplot as plt
from torchvision.models import ResNet50_Weights, ResNet18_Weights
import torch.nn.functional as F
import numpy as np
from scipy import signal

class CustomResNet(nn.Module):
    def __init__(self, version="18", num_classes=2):
        super(CustomResNet, self).__init__()
        
        # Define the first conv layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Load pre-trained ResNet (for other layers)
        if version == "18":
            resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            
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
    nrows = quadmesh.get_coordinates().shape[0] 
    ncols = quadmesh.get_coordinates().shape[1] 
    

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

if __name__ == "__main__":
    # function to transform an epoch of acceelration of the shearr buidling
    def transform_epoch_shearbuilding(epoch):
        # calculate the spectrogram
        fs = 4096
        f, t_spec, Sxx = signal.spectrogram(epoch, fs)
        
        # select subplot
        quadmesh = plt.pcolormesh(t_spec, f, 10*np.log10(Sxx), shading = 'gouraud')
        return pcolormesh_to_array(quadmesh)

    # Create the dataset and dataloader
    dataset = ShearBuildingLoader(transform_epoch_shearbuilding)
    dataset.get_data_instances(16384, 5) # 4 seconds epochs since sample_rate = 4096 and 16384 = 4096 * 4
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize the model and optimizer
    model = CustomResNet(version="50", num_classes=2).double()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epoch_losses = []
    epoch_accuracies = []

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch in dataloader:
            inputs, labels = batch

            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item()

            # Get predictions from the maximum value
            _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        # Calculate average loss and accuracy
        average_loss = running_loss / len(dataloader)
        accuracy = 100 * correct_predictions / total_predictions
        
        # Store the loss and accuracy
        epoch_losses.append(average_loss)
        epoch_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Plotting accuracy as function of epoch
    plt.plot(epoch_accuracies)
    plt.title('Accuracy as function of epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.show()