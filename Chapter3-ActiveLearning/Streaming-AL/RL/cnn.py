from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from dataloader import ShearBuildingLoader, BuildingLoader, LUMODataset
import torch
from torchviz import make_dot
from dataloader import Z24Loader
import matplotlib.pyplot as plt
from torchvision.models import ResNet50_Weights, ResNet18_Weights
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np
from scipy import signal

class CustomResNet(nn.Module):
    def __init__(self, version="18", num_classes=2):
        super(CustomResNet, self).__init__()
        
        # Define the first conv layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1)
        
        # Load pre-trained ResNet (for other layers)
        if version == "18":
            resnet = models.resnet18(weights = ResNet18_Weights.DEFAULT)
            final_conv_out_size = 512
        else:
            resnet = models.resnet50(weights = ResNet50_Weights.DEFAULT)
            final_conv_out_size = 512 * 4  # because of the expanded ResNet-50 bottleneck blocks
            
        # Replace the first convolutional layer in the pre-trained model with our 5-channel one
        resnet.conv1 = self.conv1
        
        # Use all layers of the ResNet model, but don't include the fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Calculate dense layers dynamically
        dense_layers = []
        current_input_size = final_conv_out_size
        while current_input_size > 32:
            next_output_size = current_input_size // 2
            dense_layers.append(nn.Linear(current_input_size, next_output_size))
            dense_layers.append(nn.ReLU())  # add an activation function
            current_input_size = next_output_size

        # Add the final layer to output `num_classes`
        dense_layers.append(nn.Linear(current_input_size, num_classes))
        self.dense = nn.Sequential(*dense_layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x
    
    def feature_vec(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
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

def get_accuracy(model, data_loader, device):
        correct = 0
        total = 0
        model.eval()
        with torch.inference_mode():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    # function to transform an epoch of acceelration of the shear buidling or z24 
def transform_epoch(epoch, fs):
    # calculate the spectrogram
    f, t_spec, Sxx = signal.spectrogram(epoch, fs)
    
    # select subplot
    quadmesh = plt.pcolormesh(t_spec, f, 10*np.log10(Sxx), shading = 'gouraud')
    return pcolormesh_to_array(quadmesh)

if __name__ == "__main__":
    building_type = 2 # set to 0 for shear loader, for Z24 set to 1, 2 for LUMO
    
    # size of each epoch (continuous segment/chunk of samples)
    z24_epoch_size = 16384
    shear_epoch_size = 2048 # should probably be smaller for shear since we have less data there
    building_epoch_size = 16384
    lumo_epoch_size = 16384 * (1651/100) # same 2.7 min as for Z24 but since SR is greater for LUMO than Z24 we need to increase accordingly

    z24_fs = 100 # sampling rate for z24
    shear_fs = 4096 #  .. and for shear building
    building_fs = 200
    lumo_fs = 1651

    # Create the dataset and dataloader
    dataset_train = ShearBuildingLoader(shear_epoch_size, lambda epoch: transform_epoch(epoch, shear_fs)) if \
        building_type == 0 else  Z24Loader(z24_epoch_size, lambda epoch: transform_epoch(epoch, z24_fs)) if \
        building_type == 1 else LUMODataset(lumo_epoch_size, lambda epoch: transform_epoch(epoch, lumo_fs)) if building_type == 2 else None
    dataset_train.get_data_instances(0, 1) 
    print(f"Training on {dataset_train.instances.shape[0]} instances")

    dataset_test = ShearBuildingLoader(shear_epoch_size, lambda epoch: transform_epoch(epoch, shear_fs)) if \
        building_type == 0 else Z24Loader(z24_epoch_size, lambda epoch: transform_epoch(epoch, z24_fs)) if \
        building_type == 1 else LUMODataset(lumo_epoch_size, lambda epoch: transform_epoch(epoch, lumo_fs)) if building_type == 2 else None
    dataset_test.get_data_instances(1, 1) # 4 seconds epochs since sample_rate = 4096 and 16384 = 4096 * 4

    torch.cuda.empty_cache()
    train_dataloader = DataLoader(dataset_train, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(dataset_test, batch_size=4, shuffle=True)

    # visualize one training instance
    visualize_instance = True # set to true to visualize 

    if visualize_instance:
        img_data, label = dataset_train.__getitem__(4)
        print(f"Each instance has size {img_data.shape}") # each instance of pcolormesh image is of size (3, 129, 73) -> (3, 645, 73)
        img = img_data.transpose(1, 2, 0)
        print(img)

        # Display the image
        plt.imshow(img)
        plt.axis('off')  # To turn off axis numbers
        plt.show()

    # Initialize the model and optimizer
    model = CustomResNet(version="50", num_classes=2).double()
    #model.load_state_dict(torch.load('model_weights.pth'))

    # to visualize computation graph
    """
    dummy_input = torch.randn(1, 3, 645, 73).double()
    output = model(dummy_input)

    # Visualize the model
    dot = make_dot(output)
    dot.render("model_architecture", format="png")
    """

    print(f"CUDA availability {torch.cuda.is_available()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # After 3 epochs, the learning rate will be multiplied by 0.5 (i.e., 0.001 * 0.5 = 0.0005)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
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
        
        test_accuracy = 100*get_accuracy(model, test_dataloader, device) 
        test_accuracies.append(test_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Test Accuracy: {test_accuracy:.2f}%")

        # Calculate average loss and accuracy
        average_loss = running_loss / len(train_dataloader)
        accuracy = 100 * correct_predictions / total_predictions
        
        # Store the loss and accuracy
        epoch_losses.append(average_loss)
        epoch_accuracies.append(accuracy)
        scheduler.step()

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