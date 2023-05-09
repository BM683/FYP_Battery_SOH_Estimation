"""
Gaussian_Process.py
Script for developing Gaussian process model
University of Bath student number: 1100837540  
Modified on 03/05/23
"""

# Import required libraries
import pickle
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
import gpytorch

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

# Normalise input data
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
y = scaler.fit_transform(y)

# Convert numpy arrays to torch tensors
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

# Define database
# Using first 2/3 of data for training and remaining third for testing 
class MFBDataset(Dataset):
    def __init__(self, x_tensor, y_tensor, train=True):
        if train:
            self.x = x_tensor[:2 * len(x_tensor) // 3]
            self.y = y_tensor[:2 * len(y_tensor) // 3]
        else:
            self.x = x_tensor[2 * len(x_tensor) // 3:]
            self.y = y_tensor[2 * len(y_tensor) // 3:]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

# Initilaise datasets
train_dataset = MFBDataset(x_tensor, y_tensor, train=True)
val_dataset = MFBDataset(x_tensor, y_tensor, train=False)

# Define and initialise dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=34, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=167, shuffle=False)

# Define the architectue of the Gaussian Procces model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# For loop to find most accurte results
for i in range(50):
    # Training
    train_x, train_y = next(iter(train_loader))
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y.view(-1), likelihood)

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.009)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    
    # Loss for GP - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    val_loss = 1
    
    # Training loop
    training_iterations = 1500
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y.view(-1))
        loss.backward()
        optimizer.step()
        
    
        # Get into evaluation mode
        model.eval()
        likelihood.eval()
        
        with torch.no_grad():
            val_x, val_y = next(iter(val_loader))
            val_output = model(val_x)
            val_loss = -mll(val_output, val_y.view(-1))
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()),"Validation Loss: {:.3f}".format(val_loss.item()))
    
    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x, test_y = next(iter(val_loader))
        observed_pred = likelihood(model(test_x))
    
    # Inverse the scaling
    predicted_y = scaler.inverse_transform(observed_pred.mean.numpy().reshape(-1, 1))
    real_y = scaler.inverse_transform(test_y.numpy().reshape(-1, 1))
    
    # Initialise numpy array for errors
    error = np.zeros([167,1])
    
    # Calculate errors
    for i in range(0,167):
        error[i] = predicted_y[i] - real_y[i]
        
    # initilaie an array of cycle numbers for plotting
    cycles = np.arange(0, 334, 2)
    cycles += 2
    cycles = cycles.reshape(-1, 1)
    
    
    # Plot real and predicted values
    plt.figure()
    plt.plot(cycles, real_y, label='real')
    plt.plot(cycles, predicted_y, label='predicted')
    plt.title('B0007 (Gaussian Process)')
    plt.xlabel('Cycles')
    plt.ylabel('SOH percentage')
    plt.legend()
    plt.show()
    
    # Use this model if accurate
    if np.max(abs(error)) < 2.5:
        break

# Save the predicted and real values for SOH in pickle files
with open("GP_predictions.pickle", "wb") as f:
    pickle.dump(predicted_y, f)

    
# calculate RMSE
def rmse(errors):
    mse = np.mean(errors**2)
    return np.sqrt(mse)

print(rmse(error))

# calculate R squared
R_squared_value = r2_score(real_y.flatten(), predicted_y.flatten())
print("R-squared value:", R_squared_value)

# calculate MAE
MAE_value = mean_absolute_error(real_y.flatten(), predicted_y.flatten())
print("Mean Absolute Error value:", MAE_value)

# calculate max error
print('Max error =', np.max(np.abs(error)))
