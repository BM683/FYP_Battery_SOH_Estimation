"""
Structure1_DNN_Optim.py
Script for running optimisation on the structure 1 DNN architecture
University of Bath student number: 1100837540  
Modified on 10/05/23
"""

# Import required libraries
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import  MinMaxScaler



# load the lists from the file using pickle
with open("features.pickle", "rb") as f:
    data = pickle.load(f)
    
# Remove rows for voltage, current and temp
data = np.delete(data, [0,2,4,6,8,10], axis=1)


# Clean outliers in data by averaging previous and next, keeping the same binary identifier
binary_22 = data[22,5]
binary_356 = data[356,5]
binary_690 = data[690,5]

data[22,:] = (data[21,:] + data[23,:])/2
data[356,:] = (data[355,:] + data[357,:])/2
data[690,:] = (data[689,:] + data[691,:])/2

data[22,5] = binary_22
data[356,5] = binary_356
data[690,5] = binary_690

# Define x and y data
x = data[:, :-1]
y = data[:, -1].reshape(-1, 1)

# Define hyperparameters
input_size = 6
hidden_size = 10
num_classes = 1
num_epochs = 519
batch_size = 34
learning_rate = 0.01



# Define database
# Using first 2/3 of data for training and remaining third for testing 
class MFBDataset(Dataset):
    def __init__(self, x_tensor, y_tensor, train=True):
        if train:
            self.x = x_tensor[:2*len(x_tensor)//3]
            self.y = y_tensor[:2*len(y_tensor)//3]
        else:
            self.x = x_tensor[2*len(x_tensor)//3:]
            self.y = y_tensor[2*len(y_tensor)//3:]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

# Define the architectue of the DNN
# Comment out a layer to reduce the number of hidden layers, currently there
# are three hidden layers
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        # self.fc4 = nn.Linear(hidden_size, hidden_size)
        # self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        # out = self.fc4(out)
        # out = self.relu4(out)
        out = self.fc5(out)
        return out

# Normalise data for x
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# Convert to pytorch tensors
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

# Initialise datasets
train_dataset = MFBDataset(x_tensor, y_tensor, train=True)
val_dataset = MFBDataset(x_tensor, y_tensor, train=False)


# Initialise list to save the RMSE values
rmse_list = []

# Define the list for potential number of neurons
hiddens = [2,4,8,10,15,25,40]


# Iterate through list if neurons to get rmse for each
for i in hiddens:
    hidden_size = i

    # Set the random seed for reproducibility
    torch.manual_seed(42)
    
    # Define and initialise dataloaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=334, shuffle=False)
    
    
    # Initialise model
    model = NeuralNet(input_size, num_classes)
    
    # loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Lists to keep track of losses during training
    train_losses = []
    val_losses = []
    
    # Train the model
    for epoch in range(num_epochs):
        
        # Training
        model.train()
        train_loss = 0
        for i, (x_train, y_train) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(x_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))
    
        # Validation
        model.eval()
        val_loss = 0
        val_output = []
        real_output = []
        with torch.no_grad():
            for i, (x_val, y_val) in enumerate(val_loader):
                output = model(x_val)
                val_output.append(output)
                real_output.append(y_val)
                loss = criterion(output, y_val)
                val_loss += loss.item()
            val_losses.append(val_loss / len(val_loader))
        # Print loss
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
    
    # convert to numpy arrays and concatenate horizontally
    real_y = real_output[0].numpy()
    predicted_y = val_output[0].numpy()
    
    # Apply the filter to the data
    window_size = 4
    smooth_data = np.convolve(predicted_y.flatten(), np.ones(window_size)/window_size, mode='same')
    smooth_data[:2] = 100
    smooth_data[332:]= smooth_data[330]
    
    # Initialise numpy arrays for errors
    raw_error = np.zeros([334,1])
    smooth_error = np.zeros([334,1])
    
    # Calculate errors
    for i in range(0,334):
        raw_error[i] = predicted_y[i] - real_y[i]
        smooth_error[i] = smooth_data[i] - real_y[i]
    
    # define rmse function
    def rmse(errors):
        mse = np.mean(errors**2)
        return np.sqrt(mse)
    
    # Add the rmse to the list
    rmse_list.append(rmse(smooth_error))
    
