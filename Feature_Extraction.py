"""
Feature_extraction.py
Script for extracting features in data
University of Bath student number: 1100837540  
Modified on 25/04/23
"""

# Import required libraries
import numpy as np
import pickle

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
##--------------------B0005--------------------
# Load in charge data from battery 5
with open('B0005_charge.pickle', 'rb') as f:
    B0005_charge = pickle.load(f)
    
# Remove incorrect data
B0005_charge.pop("615")
B0005_charge.pop("84")
B0005_charge.pop("0")

# Load in discharge data from battery 5
with open('B0005_discharge.pickle', 'rb') as f:
    B0005_discharge = pickle.load(f)
    
# Remove incorrect data
B0005_discharge.pop("613")  

    
###############################################################################
###############################################################################
##----------CHARGE----------
# Create an array (nx3) displaying cycle number, time of max terminal voltage and that max voltage
charge_terminal_voltage_B0005 = np.empty((len(B0005_charge),3))
count = 0
for cycle in B0005_charge.keys():
    
    voltage_output_values = B0005_charge[cycle]['voltage_battery']
    threshold = 4.18
    max_voltage_idx = None
    for i in range(len(voltage_output_values)):
        if voltage_output_values[i] > threshold:
            max_voltage_idx = i
            break

    max_voltage_time = B0005_charge[cycle]['time'][max_voltage_idx]
    max_voltage = voltage_output_values[max_voltage_idx]
    
    time_of_max_voltage = B0005_charge[cycle]['time'][max_voltage_idx]

    # Store in numpy array
    charge_terminal_voltage_B0005[count, 0] = cycle
    charge_terminal_voltage_B0005[count, 1] = time_of_max_voltage
    charge_terminal_voltage_B0005[count, 2] = 0
    
    # Iterate counter
    count += 1
    
    
# Create an array (nx3) displaying cycle number, time of when output current starts to drop and that output current.
charge_output_current_B0005 = np.empty((len(B0005_charge),3))
count = 0

for cycle in B0005_charge.keys():
    
    current_output_values = B0005_charge[cycle]['current_battery']
    start = current_output_values[0:11]
    for i in range(10):
        if current_output_values[i] < (np.average(start)-0.5):
            current_output_values[i] = start[9]
    
    threshold = 0.9 * np.max(current_output_values)
    drop_idx = np.where(current_output_values < threshold)[0][0]
    
    
    max_current_time = B0005_charge[cycle]['time'][drop_idx]
    max_current = current_output_values[drop_idx]
    
    time_of_max_current = B0005_charge[cycle]['time'][drop_idx]

    #Store in numpy array
    charge_output_current_B0005[count, 0] = cycle
    charge_output_current_B0005[count, 1] = time_of_max_current
    charge_output_current_B0005[count, 2] = 0
    count += 1
    
# Set all currents in charging to positive
charge_output_current_B0005[:,2] = np.abs(charge_output_current_B0005[:,2])


# Create an array (nx3) displaying cycle number, time of max temperature and max temperature
charge_temperature_B0005 = np.empty((len(B0005_charge),3))
count = 0
for cycle in B0005_charge.keys():
    if count < 30:
        temp_values = B0005_charge[cycle]['temp_battery']
        temp_values[:60] = np.zeros(60)
    
    else: 
        temp_values = B0005_charge[cycle]['temp_battery']
    max_temp_idx = np.argmax(temp_values)
    max_temp_time = B0005_charge[cycle]['time'][max_temp_idx]
    max_temp = temp_values[max_temp_idx]
    
    time_of_max_temp = B0005_charge[cycle]['time'][max_temp_idx]

    # Store values in numpy array
    charge_temperature_B0005[count, 0] = cycle
    charge_temperature_B0005[count, 1] = time_of_max_temp
    charge_temperature_B0005[count, 2] = max_temp
    
    # Itearte counter
    count += 1

# Create an array (nx3) displaying cycle number, time of when measured current starts to drop and that mesured current.
charge_load_current_B0005 = np.empty((len(B0005_charge),3))
count = 0
for cycle in B0005_charge.keys():
    
    current_load_values = B0005_charge[cycle]['current_load']
    
    start = current_load_values[0:11]
    for i in range(10):
        if current_load_values[i] < (np.average(start)-0.5):
            current_load_values[i] = start[9]
    
    threshold = 0.95 * np.max(current_load_values)
    
    drop_idx = np.where(current_load_values < threshold)[0][0]
    
    max_current_time = B0005_charge[cycle]['time'][drop_idx]
    max_current = current_load_values[drop_idx]
    
    time_of_max_current = B0005_charge[cycle]['time'][drop_idx]

    # Store values in numpy array
    charge_load_current_B0005[count, 0] = cycle
    charge_load_current_B0005[count, 1] = time_of_max_current
    charge_load_current_B0005[count, 2] = 0
    
    # Iterate counter
    count += 1
    
# Set all currents in charging to positive
charge_load_current_B0005[:,2] = np.abs(charge_load_current_B0005[:,2])
    
# Create an array (nx3) displaying cycle number, time of max voltage and max voltage
charge_load_voltage_B0005 = np.empty((len(B0005_charge),5))
count = 0
for cycle in B0005_charge.keys():
    
    voltage_load_values = B0005_charge[cycle]['voltage_load']
    max_voltage_idx = np.argmax(voltage_load_values)
    max_voltage_time = B0005_charge[cycle]['time'][max_voltage_idx]
    max_voltage = voltage_load_values[max_voltage_idx]
    
    time_of_max_voltage = B0005_charge[cycle]['time'][max_voltage_idx]

    charge_load_voltage_B0005[count, 0] = cycle
    charge_load_voltage_B0005[count, 1] = time_of_max_voltage
    charge_load_voltage_B0005[count, 2] = 0
     
    count += 1
    
charge_load_voltage_B0005[:, 3] = 0 
charge_load_voltage_B0005[:, 4] = 0

###############################################################################
###############################################################################
## ----------DISCHARGE----------


# Create an array (nx3) displaying cycle number, time of min terminal voltage and that min terminal voltage
discharge_terminal_voltage_B0005 = np.empty((len(B0005_discharge),3))
count = 0
for cycle in B0005_discharge.keys():
    
    voltage_load_values = B0005_discharge[cycle]['voltage_battery']
    max_voltage_idx = np.argmin(voltage_load_values)
    max_voltage_time = B0005_discharge[cycle]['time'][max_voltage_idx]
    max_voltage = voltage_load_values[max_voltage_idx]
    
    time_of_max_voltage = B0005_discharge[cycle]['time'][max_voltage_idx]

    discharge_terminal_voltage_B0005[count, 0] = cycle
    discharge_terminal_voltage_B0005[count, 1] = time_of_max_voltage
    discharge_terminal_voltage_B0005[count, 2] = 0
    
    count += 1
    
# Create an array (nx3) displaying cycle number, time of when output current starts to rise and that output current.
discharge_output_current_B0005 = np.empty((len(B0005_discharge),3))
count = 0
for cycle in B0005_discharge.keys():
    
    rise_idx = 999
    current_output_values = B0005_discharge[cycle]['current_battery']
    
    count_inner = 0
    for i in current_output_values:
        #print(count)
        #print(abs(i - current_load_values[count_inner]) )
        if count_inner < 10:
            count_inner = count_inner + 1
            
        else:
            
            if abs(i - current_output_values[count_inner]) > 1.5:
                rise_idx = count_inner 
                break
            else:
                count_inner = count_inner + 1
    if rise_idx == 999:
        rise_idx = len(current_output_values)-1
    
    max_current_time = B0005_discharge[cycle]['time'][rise_idx-1]
    max_current = current_output_values[rise_idx-1]
    
    time_of_max_current = B0005_discharge[cycle]['time'][rise_idx-1]

    discharge_output_current_B0005[count, 0] = cycle
    discharge_output_current_B0005[count, 1] = time_of_max_current
    discharge_output_current_B0005[count, 2] = 0
    
    count += 1

# Set all currents in charging to positive
discharge_output_current_B0005[:,2] = np.abs(discharge_output_current_B0005[:,2])


# Create an array (nx3) displaying cycle number, time of max temperature and max temperature
discharge_temperature_B0005 = np.empty((len(B0005_discharge),3))
count = 0
for cycle in B0005_discharge.keys():
    
    temp_values = B0005_discharge[cycle]['temp_battery']
    max_temp_idx = np.argmax(temp_values)
    max_temp_time = B0005_discharge[cycle]['time'][max_temp_idx]
    max_temp = temp_values[max_temp_idx]
    
    time_of_max_temp = B0005_discharge[cycle]['time'][max_temp_idx]

    discharge_temperature_B0005[count, 0] = cycle
    discharge_temperature_B0005[count, 1] = time_of_max_temp
    discharge_temperature_B0005[count, 2] = max_temp
    
    count += 1
    
# Create an array (nx3) displaying cycle number, time of when measured current starts to rise and that mesured current.
discharge_load_current_B0005 = np.empty((len(B0005_discharge),3))
count = 0

for cycle in B0005_discharge.keys():
    rise_idx = 999
    current_load_values = B0005_discharge[cycle]['current_load']
    
    count_inner = 0
    for i in current_load_values:
        if count_inner < 10:
            count_inner = count_inner + 1
            
        else:
            
            if abs(i - current_load_values[count_inner]) > 1:
                rise_idx = count_inner 
                break
            else:
                count_inner = count_inner + 1
    if rise_idx == 999:
        rise_idx = len(current_load_values)-1
    
    
    max_current = current_load_values[rise_idx-1]
    time_of_max_current = B0005_discharge[cycle]['time'][rise_idx]

    discharge_load_current_B0005[count, 0] = cycle
    discharge_load_current_B0005[count, 1] = time_of_max_current
    discharge_load_current_B0005[count, 2] = 0
    
    count += 1
    
# Set all currents in charging to positve
discharge_load_current_B0005[:,2] = np.abs(discharge_load_current_B0005[:,2])
    
# Create an array (nx3) displaying cycle number, time of min voltage and that min voltage
discharge_load_voltage_B0005 = np.empty((len(B0005_discharge),5))
count = 0
for cycle in B0005_discharge.keys():
    
    voltage_load_values = B0005_discharge[cycle]['voltage_load']
    threshold = 0.1 + np.min((voltage_load_values))
    idx = np.where(voltage_load_values < threshold)[0]
    
    for i in idx:
        if ( i > 30 ):
            min_idx = i
            break
    
    
    max_voltage_idx = np.argmax(voltage_load_values)
    max_voltage_time = B0005_discharge[cycle]['time'][min_idx-1]
    max_voltage = voltage_load_values[min_idx-1]
    
    time_of_max_voltage = B0005_discharge[cycle]['time'][min_idx-1]

    discharge_load_voltage_B0005[count, 0] = cycle
    discharge_load_voltage_B0005[count, 1] = time_of_max_voltage
    discharge_load_voltage_B0005[count, 2] = 0
    
    count += 1
    
discharge_load_voltage_B0005[:, 3] = 1     

    
# Calculaet the SOH
capacity_values_B0005 = np.empty((len(B0005_discharge),1))
SOH_B0005 = np.empty((len(B0005_discharge),1))
count = 0
for i in B0005_discharge:
    capacity_values_B0005[count] = B0005_discharge[i]['capacity']
    count = count + 1

max_capacity_B0005 = np.max(capacity_values_B0005)
count = 0
for i in capacity_values_B0005:
    SOH_B0005[count] = 100 * (capacity_values_B0005[count] / max_capacity_B0005)
    count = count + 1
    
    
discharge_load_voltage_B0005[:,4] = SOH_B0005[:,0]
    

# Combine the 5 arrays for charge and discharge
charge_B0005 = np.concatenate((charge_terminal_voltage_B0005, charge_output_current_B0005[:,1:], charge_temperature_B0005[:,1:], charge_load_current_B0005[:,1:], charge_load_voltage_B0005[:,1:]), axis=1)
discharge_B0005 = np.concatenate((discharge_terminal_voltage_B0005, discharge_output_current_B0005[:,1:], discharge_temperature_B0005[:,1:], discharge_load_current_B0005[:,1:], discharge_load_voltage_B0005[:,1:]), axis=1)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
##--------------------B0006--------------------

with open('B0006_charge.pickle', 'rb') as f:
    B0006_charge = pickle.load(f)
    
# Remove incorrect data
B0006_charge.pop("615")
B0006_charge.pop("84")
B0006_charge.pop("0")  
 
with open('B0006_discharge.pickle', 'rb') as f:
    B0006_discharge = pickle.load(f)
B0006_discharge.pop("613")  


###############################################################################
###############################################################################
##----------CHARGE----------


# Create an array (nx3) displaying cycle number, time of max voltage and max voltage
charge_terminal_voltage_B0006 = np.empty((len(B0006_charge),3))
count = 0
for cycle in B0006_charge.keys():
    
    voltage_output_values = B0006_charge[cycle]['voltage_battery']
    threshold = 4.18
    max_voltage_idx = None
    for i in range(len(voltage_output_values)):
        if voltage_output_values[i] > threshold:
            max_voltage_idx = i
            break

    max_voltage_time = B0006_charge[cycle]['time'][max_voltage_idx]
    max_voltage = voltage_output_values[max_voltage_idx]
    
    time_of_max_voltage = B0006_charge[cycle]['time'][max_voltage_idx]

    charge_terminal_voltage_B0006[count, 0] = cycle
    charge_terminal_voltage_B0006[count, 1] = time_of_max_voltage
    charge_terminal_voltage_B0006[count, 2] = 0
    
    count += 1
    
# Create an array (nx3) displaying cycle number, time of when output current starts to drop and that output current.
charge_output_current_B0006 = np.empty((len(B0006_charge),3))
count = 0
for cycle in B0006_charge.keys():
    
    current_output_values = B0006_charge[cycle]['current_battery']
    start = current_output_values[0:11]
    for i in range(10):
        if current_output_values[i] < (np.average(start)-0.5):
            current_output_values[i] = start[9]
    
    threshold = 0.9 * np.max(current_output_values)
    drop_idx = np.where(current_output_values < threshold)[0][0]

    
    max_current_time = B0006_charge[cycle]['time'][drop_idx]
    max_current = current_output_values[drop_idx]
    
    time_of_max_current = B0006_charge[cycle]['time'][drop_idx]

    charge_output_current_B0006[count, 0] = cycle
    charge_output_current_B0006[count, 1] = time_of_max_current
    charge_output_current_B0006[count, 2] = 0
    
    count += 1
    
# Set all currents in charging to positive
charge_output_current_B0006[:,2] = np.abs(charge_output_current_B0006[:,2])    


# Create an array (nx3) displaying cycle number, time of max temperature and max temperature
charge_temperature_B0006 = np.empty((len(B0006_charge),3))
count = 0
for cycle in B0006_charge.keys():
    
    if count < 30:
        temp_values = B0006_charge[cycle]['temp_battery']
        temp_values[:150] = np.zeros(150)
    
    else: 
        temp_values = B0006_charge[cycle]['temp_battery']
    max_temp_idx = np.argmax(temp_values)
    max_temp_time = B0006_charge[cycle]['time'][max_temp_idx]
    max_temp = temp_values[max_temp_idx]
    
    time_of_max_temp = B0006_charge[cycle]['time'][max_temp_idx]

    charge_temperature_B0006[count, 0] = cycle
    charge_temperature_B0006[count, 1] = time_of_max_temp
    charge_temperature_B0006[count, 2] = max_temp
    
    count += 1

# Create an array (nx3) displaying cycle number, time of when measured current starts to drop and that mesured current.
charge_load_current_B0006 = np.empty((len(B0006_charge),3))
count = 0
for cycle in B0006_charge.keys():
    
    current_load_values = B0006_charge[cycle]['current_load']
    
    start = current_load_values[0:11]
    for i in range(10):
        if current_load_values[i] < (np.average(start)-0.5):
            current_load_values[i] = start[9]
    
    threshold = 0.95 * np.max(current_load_values)
    
    drop_idx = np.where(current_load_values < threshold)[0][0]
    
    max_current_time = B0006_charge[cycle]['time'][drop_idx]
    max_current = current_load_values[drop_idx]
    
    time_of_max_current = B0006_charge[cycle]['time'][drop_idx]

    charge_load_current_B0006[count, 0] = cycle
    charge_load_current_B0006[count, 1] = time_of_max_current
    charge_load_current_B0006[count, 2] = 0
    
    count += 1
    
# Set all currents in charging to positive
charge_load_current_B0006[:,2] = np.abs(charge_load_current_B0006[:,2]) 
    
# Create an array (nx3) displaying cycle number, time of max voltage and max voltage
charge_load_voltage_B0006 = np.empty((len(B0006_charge),5))
count = 0
for cycle in B0006_charge.keys():
    
    voltage_load_values = B0006_charge[cycle]['voltage_load']
    max_voltage_idx = np.argmax(voltage_load_values)
    max_voltage_time = B0006_charge[cycle]['time'][max_voltage_idx]
    max_voltage = voltage_load_values[max_voltage_idx]
    
    time_of_max_voltage = B0006_charge[cycle]['time'][max_voltage_idx]

    charge_load_voltage_B0006[count, 0] = cycle
    charge_load_voltage_B0006[count, 1] = time_of_max_voltage
    charge_load_voltage_B0006[count, 2] = 0
    
    count += 1
    
charge_load_voltage_B0006[:, 3] = 0
charge_load_voltage_B0006[:, 4] = 0
    
###############################################################################
###############################################################################
## ----------DISCHARGE----------


# Create an array (nx3) displaying cycle number, time of min voltage and min voltage
discharge_terminal_voltage_B0006 = np.empty((len(B0006_discharge),3))
count = 0
for cycle in B0006_discharge.keys():
    
    voltage_load_values = B0006_discharge[cycle]['voltage_battery']
    max_voltage_idx = np.argmin(voltage_load_values)
    max_voltage_time = B0006_discharge[cycle]['time'][max_voltage_idx]
    max_voltage = voltage_load_values[max_voltage_idx]
    
    time_of_max_voltage = B0006_discharge[cycle]['time'][max_voltage_idx]

    discharge_terminal_voltage_B0006[count, 0] = cycle
    discharge_terminal_voltage_B0006[count, 1] = time_of_max_voltage
    discharge_terminal_voltage_B0006[count, 2] = 0
    
    count += 1
    
# Create an array (nx3) displaying cycle number, time of when output current starts to rise and that output current.
discharge_output_current_B0006 = np.empty((len(B0006_discharge),3))
count = 0
for cycle in B0006_discharge.keys():
    
    rise_idx = 999
    current_output_values = B0006_discharge[cycle]['current_battery']
    
    count_inner = 0
    for i in current_output_values:
        #print(count)
        #print(abs(i - current_load_values[count_inner]) )
        if count_inner < 10:
            count_inner = count_inner + 1
            
        else:
            
            if abs(i - current_output_values[count_inner]) > 1.5:
                rise_idx = count_inner 
                break
            else:
                count_inner = count_inner + 1
    if rise_idx == 999:
        rise_idx = len(current_output_values)-1
      
    max_current_time = B0006_discharge[cycle]['time'][rise_idx]
    max_current = current_output_values[rise_idx-1]
    
    time_of_max_current = B0006_discharge[cycle]['time'][rise_idx]

    discharge_output_current_B0006[count, 0] = cycle
    discharge_output_current_B0006[count, 1] = time_of_max_current
    discharge_output_current_B0006[count, 2] = 0
    
    count += 1

# Set all currents in charging to positive
discharge_output_current_B0006[:,2] = np.abs(discharge_output_current_B0006[:,2])

# Create an array (nx3) displaying cycle number, time of max temperature and max temperature
discharge_temperature_B0006 = np.empty((len(B0006_discharge),3))
count = 0
for cycle in B0006_discharge.keys():
    
    temp_values = B0006_discharge[cycle]['temp_battery']
    max_temp_idx = np.argmax(temp_values)
    max_temp_time = B0006_discharge[cycle]['time'][max_temp_idx]
    max_temp = temp_values[max_temp_idx]
    
    time_of_max_temp = B0006_discharge[cycle]['time'][max_temp_idx]

    discharge_temperature_B0006[count, 0] = cycle
    discharge_temperature_B0006[count, 1] = time_of_max_temp
    discharge_temperature_B0006[count, 2] = max_temp
    
    count += 1
    
# Create an array (nx3) displaying cycle number, time of when measured current starts to rise and that mesured current.
discharge_load_current_B0006 = np.empty((len(B0006_discharge),3))
count = 0

for cycle in B0006_discharge.keys():
    rise_idx = 999
    current_load_values = B0006_discharge[cycle]['current_load']
    
    count_inner = 0
    for i in current_load_values:
        #print(count)
        #print(abs(i - current_load_values[count_inner]) )
        if count_inner < 10:
            count_inner = count_inner + 1
            
        else:
            
            if abs(i - current_load_values[count_inner]) > 1:
                rise_idx = count_inner 
                break
            else:
                count_inner = count_inner + 1
    if rise_idx == 999:
        rise_idx = len(current_load_values)-1
    
    max_current = current_load_values[rise_idx-1]
    time_of_max_current = B0006_discharge[cycle]['time'][rise_idx-1]

    discharge_load_current_B0006[count, 0] = cycle
    discharge_load_current_B0006[count, 1] = time_of_max_current
    discharge_load_current_B0006[count, 2] = 0
    
    count += 1
    
# Set all currents in charging to positive
discharge_load_current_B0006[:,2] = np.abs(discharge_load_current_B0006[:,2])    
    
# Create an array (nx3) displaying cycle number, time of min voltage and that min voltage
discharge_load_voltage_B0006 = np.empty((len(B0006_discharge),5))
count = 0
for cycle in B0006_discharge.keys():
    
    voltage_load_values = B0006_discharge[cycle]['voltage_load']
    threshold = 0.1 + np.min((voltage_load_values))
    idx = np.where(voltage_load_values < threshold)[0]
    
    if idx.size < 3:
        min_idx = len(voltage_load_values)
        
    
    for i in idx:
        if ( i > 30 ):
            min_idx = i
            break
    
    
    max_voltage_idx = np.argmax(voltage_load_values)
    max_voltage_time = B0006_discharge[cycle]['time'][min_idx-1]
    max_voltage = voltage_load_values[min_idx-1]
    
    time_of_max_voltage = B0006_discharge[cycle]['time'][min_idx-1]

    discharge_load_voltage_B0006[count, 0] = cycle
    discharge_load_voltage_B0006[count, 1] = time_of_max_voltage
    discharge_load_voltage_B0006[count, 2] = 0
    
    count += 1
   
discharge_load_voltage_B0006[:, 3] = 1 
   
# Calculaet the SOH
capacity_values_B0006 = np.empty((len(B0006_discharge),1))
SOH_B0006 = np.empty((len(B0006_discharge),1))
count = 0
for i in B0006_discharge:
    capacity_values_B0006[count] = B0006_discharge[i]['capacity']
    count = count + 1

max_capacity_B0006 = np.max(capacity_values_B0006)
count = 0
for i in capacity_values_B0006:
    SOH_B0006[count] = 100 * (capacity_values_B0006[count] / max_capacity_B0006)
    count = count + 1
    
discharge_load_voltage_B0006[:,4] = SOH_B0006[:,0]
    
    
# Concatenate charge arrays and discahrge arrays.

# Combine the 5 arrays for charge and discharge
charge_B0006 = np.concatenate((charge_terminal_voltage_B0006, charge_output_current_B0006[:,1:], charge_temperature_B0006[:,1:], charge_load_current_B0006[:,1:], charge_load_voltage_B0006[:,1:]), axis=1)
discharge_B0006 = np.concatenate((discharge_terminal_voltage_B0006, discharge_output_current_B0006[:,1:], discharge_temperature_B0006[:,1:], discharge_load_current_B0006[:,1:], discharge_load_voltage_B0006[:,1:]), axis=1)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
##--------------------B0007--------------------

with open('B0007_charge.pickle', 'rb') as f:
    B0007_charge = pickle.load(f)
    
# Remove incorrect data
B0007_charge.pop("615")
B0007_charge.pop("84")
B0007_charge.pop("0")  
 
with open('B0007_discharge.pickle', 'rb') as f:
    B0007_discharge = pickle.load(f)
B0007_discharge.pop("613")  

    
###############################################################################
###############################################################################
##----------CHARGE----------
# Create an array (nx3) displaying cycle number, time of max voltage and max voltage
charge_terminal_voltage_B0007 = np.empty((len(B0007_charge),3))
count = 0
for cycle in B0007_charge.keys():
    
    voltage_output_values = B0007_charge[cycle]['voltage_battery']
    threshold = 4.18
    max_voltage_idx = None
    for i in range(len(voltage_output_values)):
        if voltage_output_values[i] > threshold:
            max_voltage_idx = i
            break

    max_voltage_time = B0007_charge[cycle]['time'][max_voltage_idx]
    max_voltage = voltage_output_values[max_voltage_idx]
    
    time_of_max_voltage = B0007_charge[cycle]['time'][max_voltage_idx]

    charge_terminal_voltage_B0007[count, 0] = cycle
    charge_terminal_voltage_B0007[count, 1] = time_of_max_voltage
    charge_terminal_voltage_B0007[count, 2] = 0
    
    count += 1
    
# Create an array (nx3) displaying cycle number, time of when output current starts to drop and that output current.
charge_output_current_B0007 = np.empty((len(B0007_charge),3))
count = 0
for cycle in B0007_charge.keys():
    
    current_output_values = B0007_charge[cycle]['current_battery']
    start = current_output_values[0:11]
    for i in range(10):
        if current_output_values[i] < (np.average(start)-0.5):
            current_output_values[i] = start[9]
    
    threshold = 0.9 * np.max(current_output_values)
    drop_idx = np.where(current_output_values < threshold)[0][0]
    
    #for i in current_output_va
    
    max_current_time = B0007_charge[cycle]['time'][drop_idx]
    max_current = current_output_values[drop_idx]
    
    time_of_max_current = B0007_charge[cycle]['time'][drop_idx]

    charge_output_current_B0007[count, 0] = cycle
    charge_output_current_B0007[count, 1] = time_of_max_current
    charge_output_current_B0007[count, 2] = 0
    count += 1
    
# Set all currents in charging to positive
charge_output_current_B0007[:,2] = np.abs(charge_output_current_B0007[:,2])


# Create an array (nx3) displaying cycle number, time of max temperature and max temperature
charge_temperature_B0007 = np.empty((len(B0007_charge),3))
count = 0
for cycle in B0007_charge.keys():
    
    if count < 30:
        temp_values = B0007_charge[cycle]['temp_battery']
        temp_values[:150] = np.zeros(150)
    
    else: 
        temp_values = B0007_charge[cycle]['temp_battery']
    
    max_temp_idx = np.argmax(temp_values)
    max_temp_time = B0007_charge[cycle]['time'][max_temp_idx]
    max_temp = temp_values[max_temp_idx]
    
    time_of_max_temp = B0007_charge[cycle]['time'][max_temp_idx]

    charge_temperature_B0007[count, 0] = cycle
    charge_temperature_B0007[count, 1] = time_of_max_temp
    charge_temperature_B0007[count, 2] = max_temp
    
    count += 1

# Create an array (nx3) displaying cycle number, time of when measured current starts to drop and that mesured current.
charge_load_current_B0007 = np.empty((len(B0007_charge),3))
count = 0
for cycle in B0007_charge.keys():
    
    current_load_values = B0007_charge[cycle]['current_load']
    
    start = current_load_values[0:11]
    for i in range(10):
        if current_load_values[i] < (np.average(start)-0.5):
            current_load_values[i] = start[9]
    
    threshold = 0.95 * np.max(current_load_values)
    
    drop_idx = np.where(current_load_values < threshold)[0][0]
    
    max_current_time = B0007_charge[cycle]['time'][drop_idx]
    max_current = current_load_values[drop_idx]
    
    time_of_max_current = B0007_charge[cycle]['time'][drop_idx]

    charge_load_current_B0007[count, 0] = cycle
    charge_load_current_B0007[count, 1] = time_of_max_current
    charge_load_current_B0007[count, 2] = 0
    
    count += 1
    
# Set all currents in charging to positive
charge_load_current_B0007[:,2] = np.abs(charge_load_current_B0007[:,2])
    
# Create an array (nx3) displaying cycle number, time of max voltage and max voltage
charge_load_voltage_B0007 = np.empty((len(B0007_charge),5))
count = 0
for cycle in B0007_charge.keys():
    
    voltage_load_values = B0007_charge[cycle]['voltage_load']
    max_voltage_idx = np.argmax(voltage_load_values)
    max_voltage_time = B0007_charge[cycle]['time'][max_voltage_idx]
    max_voltage = voltage_load_values[max_voltage_idx]
    
    time_of_max_voltage = B0007_charge[cycle]['time'][max_voltage_idx]

    charge_load_voltage_B0007[count, 0] = cycle
    charge_load_voltage_B0007[count, 1] = time_of_max_voltage
    charge_load_voltage_B0007[count, 2] = 0
    
    count += 1
    
charge_load_voltage_B0007[:, 3] = 0
charge_load_voltage_B0007[:, 4] = 0

###############################################################################
###############################################################################
## ----------DISCHARGE----------


# Create an array (nx3) displaying cycle number, time of min voltage and min voltage
discharge_terminal_voltage_B0007 = np.empty((len(B0007_discharge),3))
count = 0
for cycle in B0007_discharge.keys():
    
    voltage_load_values = B0007_discharge[cycle]['voltage_battery']
    max_voltage_idx = np.argmin(voltage_load_values)
    max_voltage_time = B0007_discharge[cycle]['time'][max_voltage_idx]
    max_voltage = voltage_load_values[max_voltage_idx]
    
    time_of_max_voltage = B0007_discharge[cycle]['time'][max_voltage_idx]

    discharge_terminal_voltage_B0007[count, 0] = cycle
    discharge_terminal_voltage_B0007[count, 1] = time_of_max_voltage
    discharge_terminal_voltage_B0007[count, 2] = 0
    
    count += 1
    
# Create an array (nx3) displaying cycle number, time of when output current starts to fall and that output current.
discharge_output_current_B0007 = np.empty((len(B0007_discharge),3))
count = 0
for cycle in B0007_discharge.keys():
    
    rise_idx = 999
    current_output_values = B0007_discharge[cycle]['current_battery']
    
    count_inner = 0
    for i in current_output_values:
        #print(count)
        #print(abs(i - current_load_values[count_inner]) )
        if count_inner < 10:
            count_inner = count_inner + 1
            
        else:
            
            if abs(i - current_output_values[count_inner]) > 1.5:
                rise_idx = count_inner 
                break
            else:
                count_inner = count_inner + 1
    if rise_idx == 999:
        rise_idx = len(current_output_values)-1
        
    max_current_time = B0007_discharge[cycle]['time'][rise_idx-1]
    max_current = current_output_values[rise_idx-1]
    
    time_of_max_current = B0007_discharge[cycle]['time'][rise_idx-1]

    discharge_output_current_B0007[count, 0] = cycle
    discharge_output_current_B0007[count, 1] = time_of_max_current
    discharge_output_current_B0007[count, 2] = 0
    
    count += 1

# Set all currents in charging to Positive
discharge_output_current_B0007[:,2] = np.abs(discharge_output_current_B0007[:,2])


# Create an array (nx3) displaying cycle number, time of max temperature and max temperature
discharge_temperature_B0007 = np.empty((len(B0007_discharge),3))
count = 0
for cycle in B0007_discharge.keys():
    
    temp_values = B0007_discharge[cycle]['temp_battery']
    max_temp_idx = np.argmax(temp_values)
    max_temp_time = B0007_discharge[cycle]['time'][max_temp_idx]
    max_temp = temp_values[max_temp_idx]
    
    time_of_max_temp = B0007_discharge[cycle]['time'][max_temp_idx]

    discharge_temperature_B0007[count, 0] = cycle
    discharge_temperature_B0007[count, 1] = time_of_max_temp
    discharge_temperature_B0007[count, 2] = max_temp
    
    count += 1
    
# Create an array (nx3) displaying cycle number, time of when measured current starts to rise and that mesured current.
discharge_load_current_B0007 = np.empty((len(B0007_discharge),3))
count = 0

for cycle in B0007_discharge.keys():
    rise_idx = 999
    current_load_values = B0007_discharge[cycle]['current_load']
    
    count_inner = 0
    for i in current_load_values:
        #print(count)
        #print(abs(i - current_load_values[count_inner]) )
        if count_inner < 10:
            count_inner = count_inner + 1
            
        else:
            
            if abs(i - current_load_values[count_inner]) > 1:
                rise_idx = count_inner 
                break
            else:
                count_inner = count_inner + 1
    if rise_idx == 999:
        rise_idx = len(current_load_values)-1
    
    
    max_current = current_load_values[rise_idx-1]
    time_of_max_current = B0007_discharge[cycle]['time'][rise_idx]

    discharge_load_current_B0007[count, 0] = cycle
    discharge_load_current_B0007[count, 1] = time_of_max_current
    discharge_load_current_B0007[count, 2] = 0
    
    count += 1
    
# Set all currents in charging to positive
discharge_load_current_B0007[:,2] = np.abs(discharge_load_current_B0007[:,2])
    
# Create an array (nx3) displaying cycle number, time of min voltage and that min voltage
discharge_load_voltage_B0007 = np.empty((len(B0007_discharge),5))
count = 0
for cycle in B0007_discharge.keys():
    
    voltage_load_values = B0007_discharge[cycle]['voltage_load']
    threshold = 0.1 + np.min((voltage_load_values))
    idx = np.where(voltage_load_values < threshold)[0]
    
    for i in idx:
        if idx.size < 3:
            min_idx = len(voltage_load_values)
        
        if ( i > 30 ):
            min_idx = i
            break
    
    
    max_voltage_idx = np.argmax(voltage_load_values)
    max_voltage_time = B0007_discharge[cycle]['time'][min_idx-1]
    max_voltage = voltage_load_values[min_idx-1]
    
    time_of_max_voltage = B0007_discharge[cycle]['time'][min_idx-1]

    discharge_load_voltage_B0007[count, 0] = cycle
    discharge_load_voltage_B0007[count, 1] = time_of_max_voltage
    discharge_load_voltage_B0007[count, 2] = 0

    count += 1

discharge_load_voltage_B0007[:, 3] = 1

# Calculaet the SOH
capacity_values_B0007 = np.empty((len(B0007_discharge),1))
SOH_B0007 = np.empty((len(B0007_discharge),1))
count = 0
for i in B0007_discharge:
    capacity_values_B0007[count] = B0007_discharge[i]['capacity']
    count = count + 1

max_capacity_B0007 = np.max(capacity_values_B0007)
count = 0
for i in capacity_values_B0007:
    SOH_B0007[count] = 100 * (capacity_values_B0007[count] / max_capacity_B0007)
    count = count + 1
    
discharge_load_voltage_B0007[:,4] = SOH_B0007[:,0]

# Combine the 5 arrays for cahrge and discharge
charge_B0007 = np.concatenate((charge_terminal_voltage_B0007, charge_output_current_B0007[:,1:], charge_temperature_B0007[:,1:], charge_load_current_B0007[:,1:], charge_load_voltage_B0007[:,1:]), axis=1)
discharge_B0007 = np.concatenate((discharge_terminal_voltage_B0007, discharge_output_current_B0007[:,1:], discharge_temperature_B0007[:,1:], discharge_load_current_B0007[:,1:], discharge_load_voltage_B0007[:,1:]), axis=1)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

# Combine them veryically
B0005_disorder = np.concatenate((charge_B0005, discharge_B0005), axis = 0)

# Order data in terms of the cycle number
B0005 = B0005_disorder[B0005_disorder[:,0].argsort()]

# Calculate SOH for each cycle, where on charge cycles, an average of previous and after charge cycle is taken 
count = 0
for i in B0005[:-1,12]:
    if count ==0:
        B0005[count,12] = 100
        
    elif count == 333:
        B0005[count,12] = B0005[count-1,12] 
        
    elif i == 0:
        if B0005[count+1,12] ==0:
            B0005[count,12] = B0005[count-1,12] 
        
        else: 
            B0005[count,12] = (B0005[count-1,12] + B0005[count+1,12])/2
    count = count + 1

B0005[len(B0005)-1,12] = B0005[len(B0005)-2,12]

# Repeat for battery 6
B0006_disorder = np.concatenate((charge_B0006, discharge_B0006), axis = 0)
B0006 = B0006_disorder[B0006_disorder[:,0].argsort()]

count = 0
for i in B0006[:-1,12]:
    if count ==0:
        B0006[count,12] = 100
        
    elif count == 333:
        B0006[count,12] = B0006[count-1,12] 
        
    elif i == 0:
        
        
            
        if B0006[count+1,12] ==0:
            B0006[count,12] = B0006[count-1,12] 
        else: 
            B0006[count,12] = (B0006[count-1,12] + B0006[count+1,12])/2
    count = count + 1
    
B0006[len(B0006)-1,12] = B0006[len(B0006)-2,12]

# Repeat for battery 7
B0007_disorder = np.concatenate((charge_B0007, discharge_B0007), axis = 0)
B0007 = B0007_disorder[B0007_disorder[:,0].argsort()]

count = 0
for i in B0007[:-1,12]:
    if count ==0:
        B0007[count,12] = 100
        
    elif count == 333:
        B0007[count,12] = B0007[count-1,12] 
    elif i == 0:
        
        
            
        if B0007[count+1,12] ==0:
            B0007[count,12] = B0007[count-1,12] 
        else: 
            B0007[count,12] = (B0007[count-1,12] + B0007[count+1,12])/2
    count = count + 1
B0007[len(B0007)-1,12] = B0007[len(B0007)-2,12]

# Combine the data from all three battreies
battery_data = np.vstack((B0005, B0006, B0007))

# save the lists to a file using pickle
with open("features.pickle", "wb") as f:
    pickle.dump(battery_data, f)
    


