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
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Load pre-trained ResNet (for other layers)
        #if version == "18":
        #    resnet = models.resnet18(weights=None)
        #else:
        #    resnet = models.resnet50(weights=None)
            
        # Replace the first convolutional layer in the pre-trained model with our 6-channel one
        #resnet.conv1 = self.conv1
        
        # Use all layers of the ResNet model, but don't include the fully connected layer
        #self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Create a new fully connected layer
        #self.fc = nn.Linear(512 * (4 if version == "50" else 1), num_classes)


        # Conv layer: num_features convolutional units with tanh activation
        self.conv1 = nn.Conv2d(3, 6, kernel_size=10, padding=1)
        
        # 1D Conv layer: num_features convolutional units with tanh activation
        self.conv2 = nn.Conv2d(6, 3, kernel_size=5, padding=1)
        
        # Calculate the size after convolution and pooling
        # If input shape is (batch_size, num_features, seq_length), 
        # The output shape from conv2 would be (batch_size, num_features, seq_length), since we're using padding.
        # However, this calculation can be complex with larger networks, so you might need to empirically determine it.
        # For this simple model, it remains the same.
        self.flattened_size = 146880  # seq_length needs to be defined based on your input
        
        # Dense layer: 200 units with tanh activation
        self.fc1 = nn.Linear(self.flattened_size, num_classes)
    
    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.dropout(x, p=0.25)
        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc1(x))
        x = F.dropout(x, p=0.50)
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
    def get_accuracy(model, data_loader, device):
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    # function to transform an epoch of acceelration of the shearr buidling
    def transform_epoch_shearbuilding(epoch):
        # calculate the spectrogram
        fs = 4096
        f, t_spec, Sxx = signal.spectrogram(epoch, fs)
        
        # select subplot
        quadmesh = plt.pcolormesh(t_spec, f, 10*np.log10(Sxx), shading = 'gouraud')
        return pcolormesh_to_array(quadmesh)

    # Create the dataset and dataloader
    dataset_train = ShearBuildingLoader(transform_epoch_shearbuilding)
    dataset_train.get_data_instances(True, 16384, 3) # 4 seconds epochs since sample_rate = 4096 and 16384 = 4096 * 4

    dataset_test = ShearBuildingLoader(transform_epoch_shearbuilding)
    dataset_test.get_data_instances(False, 16384, 3) # 4 seconds epochs since sample_rate = 4096 and 16384 = 4096 * 4


    torch.cuda.empty_cache()
    train_dataloader = DataLoader(dataset_train, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(dataset_test, batch_size=4, shuffle=True)
    img_data, label = dataset_train.__getitem__(4)
    print(f"Each instance has size {img_data.shape}")

    img = img_data.transpose(1, 2, 0)

    # Display the image
    plt.imshow(img)
    plt.axis('off')  # To turn off axis numbers
    plt.show()


    # Initialize the model and optimizer
    model = CustomResNet(version="18", num_classes=2).double()
    print("CUDA availability")
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.to(device)

    epoch_losses = []
    epoch_accuracies = [] # training accuracies
    test_accuracies = [] # test accuracies
    

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch in train_dataloader:
            inputs, labels = batch

            inputs = inputs.to(device)
            labels = labels.to(device)

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
        
        test_accuracy = get_accuracy(model, test_dataloader, device) 
        test_accuracies.append(test_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Test Accuracy: {test_accuracy:.2f}%")

        # Calculate average loss and accuracy
        average_loss = running_loss / len(train_dataloader)
        accuracy = 100 * correct_predictions / total_predictions
        
        # Store the loss and accuracy
        epoch_losses.append(average_loss)
        epoch_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), 'model_weights.pth')

    # Plotting accuracy as function of epoch
    plt.clf() # to clear 
    fig, ax = plt.subplots()
    ax.plot(np.arange(num_epochs), epoch_accuracies, label='train', color='blue')
    ax.plot(np.arange(num_epochs), test_accuracies,  label='test', color='red')

    # Add labels, title, legend, and display the plot
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Epoch')
    ax.legend()
    ax.grid(True)
    plt.show()