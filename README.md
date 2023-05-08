# FYP_Battery_SOH_Estimation

This repository includes all the python scripts that were developed for my final year project: 

**Data-Driven Estimation of Lithium-ion Battery State of Health (SOH)**


## Files included:
- Mat_Pickle_Converter.py - Python script that converts the .mat files for each battery to well structured .pickle files for charge, discharge and impedance cycles for each given battery.
- Feature_Extraction.py - Python script that extracts features from batteries 5,6 and 7 pickle files genearted from Mat_Pickle_Converter.py. Generating a .pickle file containing all features.
- Structure1_DNN.py - Python script that restructures the data to structure 1 and develops, trains and tests the corresponding DNN.
- Structure2_DNN.py - Python script that restructures the data to structure 2 and develops, trains and tests the corresponding DNN.
- Coupled_DNN.py - Python script that restructures the data to coupled structure and develops, trains and tests the corresponding DNNs.
- Gaussian_Process.py - Python script that restructures the data to structure 2 and develops, trains and test the Gaussian Process model.
- Support_Vector_Machine.py - Python script that restructures the data to structure 2 and develops, trains and test the Support vector machine model.
