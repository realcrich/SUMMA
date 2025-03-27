# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 15:34:20 2025

@author: Collin
"""

#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
import torch.utils.data as data
from SUMMA.SCRIPTS import *

def slice_df_by_date(df, date_col, start_year, start_month, start_day, end_year, end_month, end_day):
    """
    Slice a DataFrame based on a start and end date.

    :param df: Pandas DataFrame containing a timestamp column.
    :param date_col: Name of the timestamp column (string).
    :param start_year: Start year (int).
    :param start_month: Start month (int).
    :param start_day: Start day (int).
    :param end_year: End year (int).
    :param end_month: End month (int).
    :param end_day: End day (int).
    :return: Sliced DataFrame between start and end dates.
    """
    # Convert the column to datetime if it's not already
    df[date_col] = pd.to_datetime(df[date_col])

    # Define start and end timestamps
    start_date = pd.Timestamp(year=start_year, month=start_month, day=start_day)
    end_date = pd.Timestamp(year=end_year, month=end_month, day=end_day)

    # Slice the DataFrame based on the date range
    return df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]

def calculate_vpd(temp_K, RH):
    """Compute Vapor Pressure Deficit (VPD) using temperature in Kelvin and RH."""
    temp_C = temp_K - 273.15  # Convert Kelvin to Celsius
    es = 0.6108 * np.exp((17.27 * temp_C) / (temp_C + 237.3))  # Saturation vapor pressure (kPa)
    ea = es * (RH / 100)  # Actual vapor pressure (kPa)
    return es - ea  # VPD in kPa

# Define function to turn dataframe into PyTorch tensors
def create_torch_set(input_data, target_data, lookback): 
    """
    Convert dataframes for input features and output target to PyTorch Tensors to use in LSTM.

    :param data: dataframe with input features (pd.DataFrame).
    :param lookback: define how many timestamps to lookback (int).
    :return: PyTorch tensors for X and y, input features and output targets, respectively.
    """
    X, y = [], []
    
    for i in range(len(target_data)):
        if i * lookback + lookback > len(input_data):
            break  # Ensure valid indexing
        X.append(input_data.iloc[i * lookback : i * lookback + lookback].values)
        y.append(target_data.iloc[i])

    # Convert lists to properly structured NumPy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

def masked_mse_loss(pred, target):
    mask = ~torch.isnan(target)  # Mask where target is NOT NaN
    if mask.sum() == 0:  # Prevent dividing by zero if all targets are NaN
        return torch.tensor(0.0, requires_grad=True)  # Return zero loss
    return func.mse_loss(pred[mask], target[mask])

# Read in SNOTEL data for training
DF_IN = pd.read_csv('C:/Users/Collin/SUMMA/SUMMA_shared/summaTestCases_3.0/testCases_data/inputData/fieldData/sagehen/forcing_sagehen_2541.csv')
DF_OUT = pd.read_csv('C:/Users/Collin/SUMMA/SUMMA_shared/sapflux/DailyTotals_922_T4N.csv')

df_in = slice_df_by_date(DF_IN, 'timestamp', 2016, 2, 1, 2019, 9, 30)
df_out = slice_df_by_date(DF_OUT, 'Date', 2016, 2, 1, 2019, 9, 30)

nt = len(df_in)//24
df_in = df_in.iloc[:nt*24]
df_out = df_out.iloc[:nt]

# calculate VPD to add to input features
df_in['VPD'] = calculate_vpd(df_in['TA'],df_in['RH'])

# Drop RH and PSUM from dataframe (we are using SpecHum and Precip rate instead)
df_in = df_in.loc[:,['timestamp','TA','ISWR','VPD']]

# Convert timestamp to datetime
df_in['timestamp'] = pd.to_datetime(df_in['timestamp'])

# Filter out months October (10) through January (1)
df_in = df_in[~df_in['timestamp'].dt.month.isin([10, 11, 12, 1])]

# Define function to categorize periods
def time_period(hour):
    return 'day' if 4 <= hour < 21 else 'night'

# Create new columns for grouping
df_in['period'] = df_in['timestamp'].dt.hour.apply(time_period)
df_in['date'] = df_in['timestamp'].dt.date

# Compute mean and overwrite df_in
df_in = df_in.groupby(['date', 'period']).mean().reset_index()

# Drop unnecessary columns
df_in = df_in.drop(columns=['timestamp'], errors='ignore')

# make a copy of the dataframe without the timestamps included since they won't
# be used as an input feature to the model & ensure all input feature values 
# are dtype float32 for use with PyTorch

df_in = df_in.iloc[:, 2:5].copy().astype(np.float32)
df_out = df_out.iloc[:,1:].copy().astype(np.float32)

# Just select one sensor for now to train on 
#df_out = df_out['X10']

# Take average reading over all sensors for each timestep to yield one target value per timestamp 
df_out = df_out.mean(axis=1)

# Normalize to confine range (0,1) after averaging sensors
df_out = (df_out-df_out.min())/(df_out.max()-df_out.min())

# Double the length of df_out
df_out = df_out.reindex(np.arange(len(df_out) * 2))

# Assign original values to even indices
df_out.iloc[::2] = df_out.iloc[:len(df_out)//2].values

# Assign 0s to odd indices (nighttime)
df_out.iloc[1::2] = 0

# Reset index to maintain a clean structure
df_out.reset_index(drop=True, inplace=True)

df_out = df_out[:len(df_in)]
###################################################################################################
#### the dataframe should now be ready to be converted to PyTorch tensors for use in the model ####
###################################################################################################

# Define some model parameters here to use later
n_features = df_in.shape[1]
n_hidden = 256#2*n_features**2
n_layers = 4#2
#n_output = df_out.shape[1]

# this is arbitrary for now but select sizes for the training and testing sets
train_set_len = int(0.7*len(df_in))
test_set_len = int(len(df_in) - train_set_len)

# Define 'lookback', for us specifies how to chunk hourly data up (tentatively using 3, 8-hour chunks for 1 24-hr period)
#lookback = 8
#lookback = 24    # Try using 24 to represent assigning 24 hourly input values to 1 daily sum target output 
lookback = 2    # Here, 2 represents one day(04:00-21:00) or night(21:00-04:00) cycle

# use function above to create PyTorch tensors
#X_train, y_train = create_torch_set(df_in[:train_set_len*lookback],df_out[:train_set_len],lookback)
#X_test, y_test = create_torch_set(df_in[train_set_len*lookback:],df_out[train_set_len:],lookback)

# use function above to create PyTorch tensors
X_train, y_train = create_torch_set(df_in[:train_set_len],df_out[:train_set_len],lookback)
X_test, y_test = create_torch_set(df_in[train_set_len:],df_out[train_set_len:],lookback)

# Print training/testing set shapes, just for a check
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# Define model architecture, for now very simple using 1 LSTM block and 1 linear block 
class Model(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,
                            num_layers=num_layers,batch_first=True,dropout=0.1)#,
                            #bidirectional=True)    # add bidirectionalality due to sparse target values
        self.linear = nn.Linear(hidden_size,1)    # need to 2x hidden size for bidirectionality
    def forward(self,x):
        x,_ = self.lstm(x)
        x = self.linear(x[:,-1,:])
        return x

# Call model using specified parameters, name optimizer and loss functions, and create dataLoader
model = Model(n_features,n_hidden,n_layers).float()
opt = optim.Adam(model.parameters(), lr=0.0001)   # set lower learning rate 
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train),shuffle=True,batch_size=8)    # low batch size for comp. efficiency

# specifiy number of epochs to use for training
n_epochs = 1000

# Train model and calculate loss, print for every 100th epoch to monitor training progress
for epoch in range(n_epochs): 
    model.train()
    for X_batch, y_batch in loader:
        
        if torch.isnan(y_batch).all():  # Skip batch if all targets are NaN
            continue 
        
        y_pred = model(X_batch)
        #loss = loss_fn(y_pred, y_batch)
        loss = masked_mse_loss(y_pred, y_batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # Validation
    if epoch % 100 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        #train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        train_rmse = torch.sqrt(masked_mse_loss(y_pred, y_train))
        y_pred = model(X_test)
        #test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        test_rmse = torch.sqrt(masked_mse_loss(y_pred, y_test))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
    