"""
Coupled_DNN.py
Script for developing DNNs for coupled data structures
University of Bath student number: 1100837540  
Modified on 09/05/23
"""

# import required libraries
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score , mean_absolute_error


# load the lists from the file using pickle
with open("features.pickle", "rb") as f:
    data = pickle.load(f)

# Remove rows for voltage, current and temp
data = np.delete(data, [0,2,4,8,10], axis=1)

# Index of the column with the binary flag (0 or 1)
charge_discharge_column = 6  


# Charge = 0, discharge = 1
# Filter the data to keep only the charge cycles (rows with non-zero values in the specified column)
charge = data[data[:, charge_discharge_column] == 0]
discharge = data[data[:, charge_discharge_column] != 0]

# define x and y for charge and discharge data 
charge_x = charge[:, :-1]
charge_x = np.delete(charge_x,[3,6],axis =1)
charge_y = charge[:, -1].reshape(-1, 1)

discharge_x = discharge[:, :-1]
discharge_x = np.delete(discharge_x,[3,6],axis =1)
discharge_y = discharge[:, -1].reshape(-1, 1)

# define hyperparameters
input_size = 5
hidden_size = 10
num_classes = 1
charge_epochs = 200
discharge_epochs = 200
batch_size = 32
learning_rate = 0.01

# Set the random seed for reproducibility
torch.manual_seed(42)


# Define database for each DNN, charge and discharge
# Using first 2/3 of data for training and remaining third for testing 
class ChargeDataset(Dataset):
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
    
class DischargeDataset(Dataset):
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


# Define the architectue of each DNN
class ChargeNeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ChargeNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 8)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(8, 8)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 8)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(8, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = torch.clamp(out, max=100)  # apply clamp function
        return out

class DischargeNeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DischargeNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 8)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(8, 8)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 8)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(8, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = torch.clamp(out, max=100)  # apply clamp function
        return out



# Normalise all data
scaler = MinMaxScaler()
charge_x = scaler.fit_transform(charge_x)
discharge_x = scaler.fit_transform(discharge_x)

charge_y = scaler.fit_transform(charge_y)
discharge_y = scaler.fit_transform(discharge_y)

# convert from numoy arrays to torch tensors
charge_x_tensor = torch.from_numpy(charge_x).float()
charge_y_tensor = torch.from_numpy(charge_y).float()

discharge_x_tensor = torch.from_numpy(discharge_x).float()
discharge_y_tensor = torch.from_numpy(discharge_y).float()

# initialise datasets
charge_train_dataset = ChargeDataset(charge_x_tensor, charge_y_tensor, train=True)
charge_val_dataset = ChargeDataset(charge_x_tensor, charge_y_tensor, train=False)

discharge_train_dataset = DischargeDataset(discharge_x_tensor, discharge_y_tensor, train=True)
discharge_val_dataset = DischargeDataset(discharge_x_tensor, discharge_y_tensor, train=False)

# define and initialise dataloaders
charge_train_loader = DataLoader(dataset=charge_train_dataset, batch_size=batch_size, shuffle=True)
charge_val_loader = DataLoader(dataset=charge_val_dataset, batch_size=334, shuffle=False)

discharge_train_loader = DataLoader(dataset=discharge_train_dataset, batch_size=batch_size, shuffle=True)
discharge_val_loader = DataLoader(dataset=discharge_val_dataset, batch_size=334, shuffle=False)


# initialise models
charge_model = ChargeNeuralNet(input_size, num_classes)
discharge_model = DischargeNeuralNet(input_size, num_classes)

# initilaise and define loss function and optimiser
criterion = nn.MSELoss()
charge_optimiser = optim.Adam(charge_model.parameters(), lr=learning_rate, weight_decay=0.01)
discharge_optimiser = optim.Adam(discharge_model.parameters(), lr=learning_rate, weight_decay=0.01)





# Lists to keep track of losses during training of charge DNN
charge_train_losses = []
charge_val_losses = []
out = 0

# Train the charge model
for epoch in range(charge_epochs):
    # train
    charge_model.train()
    train_loss = 0
    for i, (x_train, y_train) in enumerate(charge_train_loader):
        charge_optimiser.zero_grad()
        output = charge_model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        charge_optimiser.step()
        train_loss += loss.item()
    charge_train_losses.append(train_loss / len(charge_train_loader))

    # Validation
    charge_model.eval()
    val_loss = 0
    charge_val_output = []
    charge_real_output = []
    with torch.no_grad():
        for i, (x_val, y_val) in enumerate(charge_val_loader):
            output = charge_model(x_val)
            charge_val_output.append(output)
            charge_real_output.append(y_val)
            loss = criterion(output, y_val)
            val_loss += loss.item()
            if loss < 0.006:
                out = 1
        charge_val_losses.append(val_loss / len(charge_val_loader))

    # Print loss
    print(f'Epoch [{epoch+1}/{charge_epochs}], Train Loss: {charge_train_losses[-1]:.4f}, Val Loss: {charge_val_losses[-1]:.4f}')

# Plot loss curves
plt.plot(charge_train_losses, label='Train')
plt.plot(charge_val_losses, label='Validation')
plt.ylim([0,0.2])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Lists to keep track of losses during training of discharge DNN
discharge_train_losses = []
discharge_val_losses = []
out = 0

for epoch in range(discharge_epochs):
    #Training
    discharge_model.train()
    train_loss = 0
    for i, (x_train, y_train) in enumerate(discharge_train_loader):
        discharge_optimiser.zero_grad()
        output = discharge_model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        discharge_optimiser.step()
        train_loss += loss.item()
    discharge_train_losses.append(train_loss / len(discharge_train_loader))

    # Validation
    discharge_model.eval()
    val_loss = 0
    discharge_val_output = []
    discharge_real_output = []
    with torch.no_grad():
        for i, (x_val, y_val) in enumerate(discharge_val_loader):
            output = discharge_model(x_val)
            discharge_val_output.append(output)
            discharge_real_output.append(y_val)
            loss = criterion(output, y_val)
            val_loss += loss.item()
            if loss < 0.001:
                out = 1
        discharge_val_losses.append(val_loss / len(discharge_val_loader))

    # Print loss
    print(f'Epoch [{epoch+1}/{discharge_epochs}], Train Loss: {discharge_train_losses[-1]:.4f}, Val Loss: {discharge_val_losses[-1]:.4f}')

# Plot loss curves
plt.plot(discharge_train_losses, label='Train')
plt.plot(discharge_val_losses, label='Validation')
plt.ylim([0,0.2])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# convert tensors to numpy arrays 
charge_predicted_y = charge_val_output[0].numpy()
discharge_predicted_y = discharge_val_output[0].numpy()

# Flatten the arrays
charge_predicted_y = charge_predicted_y.flatten()
discharge_predicted_y = discharge_predicted_y.flatten()

# reverse the normalisation
charge_predicted_y = scaler.inverse_transform(charge_predicted_y.reshape(-1, 1))
discharge_predicted_y = scaler.inverse_transform(discharge_predicted_y.reshape(-1, 1))

# create array for combined predictions
combined_predictions = np.zeros(charge_predicted_y.shape[0] + discharge_predicted_y.shape[0])

# Interleave the two lists
combined_predictions[::2] = charge_predicted_y.flatten()
combined_predictions[1::2] = discharge_predicted_y.flatten()

# Define the size of the moving window
window_size = 4

# Create a moving average filter
window = np.ones(int(window_size))/float(window_size)

# Apply the filter to the data
smooth_data = np.convolve(combined_predictions, np.ones(window_size)/window_size, mode='same')
smooth_data[:2] = 100
smooth_data[332:]= smooth_data[330]

# get real SOH values and define cycles for plotting
real_y = data[668:, 7]
cycles = np.arange(334).reshape(334, 1)
cycle_singular = np.arange(0, 334, 2).reshape(167, 1)

# initialise arrays for errors
charge_error = np.zeros([334,1])
discharge_error = np.zeros([334,1])
raw_error = np.zeros([334,1])
smooth_error = np.zeros([334,1])

# calculate errors and populate arrays
for i in range(0,334):
    raw_error[i] = combined_predictions[i] - real_y[i]
    smooth_error[i] = smooth_data[i] - real_y[i]
for i in range(0,167):
    charge_error[i] = charge_predicted_y[i] - real_y[i]
    discharge_error[i] = discharge_predicted_y[i] - real_y[i]
    
# plot real vs estimated for charge alone
plt.figure()
plt.figure(figsize=(11,7))
plt.plot(cycle_singular, charge_predicted_y, label='Charge estimation')
plt.plot(cycles, real_y, label='Real')
plt.title('Charge DNN - B0007', fontsize = 16)
plt.xlabel('Cycle number', fontsize = 16)
plt.ylabel('SOH', fontsize = 16)
plt.legend(fontsize = 14)
plt.tick_params(labelsize=14)
plt.show

# plot real vs estimated for discharge alone
plt.figure()
plt.figure(figsize=(11,7))
plt.plot(cycle_singular, discharge_predicted_y, label='Discharge estimation')
plt.plot(cycles, real_y, label='Real')
plt.title('Discharge DNN - B0007', fontsize = 16)
plt.xlabel('Cycle number', fontsize = 16)
plt.ylabel('SOH', fontsize = 16)
plt.legend(fontsize = 14)
plt.tick_params(labelsize=14)
plt.show

# plot real vs estimated for combined without smoothing
plt.figure()
plt.figure(figsize=(11,7))
plt.plot(cycles, combined_predictions, label='Combined estimation')
plt.plot(cycles, real_y, label='Real')
plt.title('Coupled DNN - B0007', fontsize = 16)
plt.xlabel('Cycle number', fontsize = 16)
plt.ylabel('SOH', fontsize = 16)
plt.legend(fontsize = 14)
plt.tick_params(labelsize=14)
plt.show
    
# plot real vs estimated for combined with smoothing
plt.figure()
plt.figure(figsize=(11,7))
plt.plot(cycles, smooth_data, label='Smoothed estimation')
plt.plot(cycles, real_y, label='Real')
plt.title('Coupled DNN - B0007', fontsize = 16)
plt.xlabel('Cycle number', fontsize = 16)
plt.ylabel('SOH', fontsize = 16)
plt.legend(fontsize = 14)
plt.tick_params(labelsize=14)
plt.show

# plot error for smooth vs unsmoothed
plt.figure()
plt.figure(figsize=(11,7))
plt.plot(cycles, raw_error, label='Estimation')
plt.plot(cycles, smooth_error, label='Smoothed estimation')
plt.title('Coupled DNN error - B0007', fontsize = 16)
plt.xlabel('Cycle number', fontsize = 16)
plt.ylabel('Percentage error', fontsize = 16)
plt.legend(fontsize = 14)
plt.tick_params(labelsize=14)

# define rmse function
def rmse(errors):
    mse = np.mean(errors**2)
    return np.sqrt(mse)

# print rmse for differnet errors
print('Unsmoothed RMSE: ',rmse(raw_error))
print('Smoothed RMSE: ',rmse(smooth_error))

# calculate r squared and print
R_squared_value = r2_score(real_y.flatten(), smooth_data.flatten())
print("R-squared value:", R_squared_value)

# calculate MAE and print
MAE_value = mean_absolute_error(real_y.flatten(), smooth_data.flatten())
print("DNN Mean Absolute Error value:", MAE_value)

# calculate max error and print
print('Max error =', np.max(np.abs(smooth_error)))

# save predictions to a pickle file
with open("2DNNs_predictions.pickle", "wb") as f:
    pickle.dump(smooth_data, f)
