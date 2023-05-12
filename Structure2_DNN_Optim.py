"""
Structure2_DNN_Optim.py
Script for running optimisation on the structure 2 DNN architecture
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
from sklearn.preprocessing import MinMaxScaler


# load the lists from the file using pickle
with open("features.pickle", "rb") as f:
    data = pickle.load(f)
    predata = data
    
# Remove colums for binary classification and cuurent, voltage, temp values
data = np.delete(data, [0,2,4,6,8,10,11], axis=1)

# Initialise x, all data except for SOH column
x = data[:, :-1]

# Initialize an empty list to store the new structured data
x_new_list = []

# Iterate through the rows of x, two rows at a time
for i in range(0, len(x), 2):
    # Stack two consecutive rows horizontally using hstack
    new_row = np.hstack((x[i], x[i+1]))
    # Append the new_row to the x_new_list
    x_new_list.append(new_row)

# Convert the list of new rows to a numpy array
x = np.array(x_new_list)

# Remove outliers
x[11,:] = (x[10,:] + x[12,:])/2
x[177,:] = (x[176,:] + x[179,:])/2
x[345,:] = (x[343,:] + x[344,:])/2

# Initialise and reshape y for SOH column
y = data[:, -1].reshape(-1, 1)

# reshape the array to combine every two rows
y = y.reshape(-1, 2, 1)

# take the mean along the second axis
y = np.mean(y, axis=1)

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

# Normalise input data
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
y = scaler.fit_transform(y)

# Convert numpy arrays to torch tensors
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

# Initilaise datasets
train_dataset = MFBDataset(x_tensor, y_tensor, train=True)
val_dataset = MFBDataset(x_tensor, y_tensor, train=False)

# define hyperparameters
input_size = 10
hidden_size = 10
num_classes = 1
num_epochs = 519
batch_size = 34
learning_rate = 0.01

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
    
    # Initialise DNN model
    model = NeuralNet(input_size, num_classes)
    
    # define loss function and optimiser
    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Lists to keep track of losses during training
    train_losses = []
    val_losses = []
    
    out = 0
    
    # Train the model
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for i, (x_train, y_train) in enumerate(train_loader):
            optimiser.zero_grad()
            output = model(x_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimiser.step()
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
    
    
    
    # convert to numpy arrays and reverse normalisation 
    real_y = real_output[0].numpy()
    real_y = scaler.inverse_transform(real_y)
    predicted_y = val_output[0].numpy()
    predicted_y = scaler.inverse_transform(predicted_y)
    
    # initiliase numpy array for error
    error = np.zeros([167,1])
    
    # calculate errors
    for i in range(0,167):
        error[i] = predicted_y[i] - real_y[i]
        
    # calculate RMSE
    def rmse(errors):
        mse = np.mean(errors**2)
        return np.sqrt(mse)
    
    # Add the rmse to the list
    rmse_list.append(rmse(error))
    
    
