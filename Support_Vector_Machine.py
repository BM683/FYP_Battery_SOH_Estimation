import numpy as np

import pickle
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error


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

# Normalise the input data
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

x = scaler_x.fit_transform(x)
y = scaler_y.fit_transform(y)


# Split the data into train and test sets
train_x = x[:334]
train_y = y[:334]
test_x = x[334:]
test_y = y[334:]


# Initialize the SVM regressor with the optimized hyperparameters
regr = svm.SVR()

# Train the SVM
regr.fit(train_x, train_y)

# Make predictions
predicted_y = regr.predict(test_x)
predicted_y = predicted_y.reshape(-1, 1)
real_y = test_y.reshape(-1, 1)

# Inverse the scaling
predicted_y = scaler_y.inverse_transform(predicted_y)
real_y = scaler_y.inverse_transform(real_y)

# initilaie an array of cycle numbers for plotting
cycles = np.arange(0, 334, 2)
cycles += 2
cycles = cycles.reshape(-1, 1)

# initiliase numpy array for error
error = np.zeros([167,1])

# calculate errors
for i in range(0,167):
    error[i] = predicted_y[i] - real_y[i]

# plot the estimations
plt.figure()
plt.figure(figsize=(11,7))
plt.plot(cycles, predicted_y, label='Estimation')
plt.plot(cycles, real_y, label='Real')
plt.title('Support vector machine - B0007', fontsize = 16)
plt.xlabel('Cycle number', fontsize = 16)
plt.ylabel('SOH', fontsize = 16)
plt.legend(fontsize = 14)
plt.tick_params(labelsize=14)
plt.show

# Save the predicted for SOH in pickle files
with open("SVM_predictions.pickle", "wb") as f:
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