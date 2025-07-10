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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read in SNOTEL data for training
DF_IN = pd.read_csv('C:/Users/Collin/SUMMA/SUMMA_shared/summaTestCases_3.0/testCases_data/inputData/fieldData/sagehen/forcing_sagehen_2124.csv')
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
#df_in = df_in.loc[:,['timestamp','TA','VW','Prate','RH','ISWR','VPD']]

# Convert timestamp to datetime
df_in['timestamp'] = pd.to_datetime(df_in['timestamp'])

# Filter out months October (10) through January (1)
df_in = df_in[~df_in['timestamp'].dt.month.isin([10, 11, 12, 1])]

# Drop unnecessary columns
#df_in = df_in.drop(columns=['timestamp'], errors='ignore')

# make a copy of the dataframe without the timestamps included since they won't
# be used as an input feature to the model & ensure all input feature values 
# are dtype float32 for use with PyTorch

#df_in = df_in.iloc[:, 2:].copy().astype(np.float32)
df_out = df_out.iloc[:,1:].copy().astype(np.float32)

# Just select one sensor for now to train on 
#df_out = df_out['X10']

#df_out['min'] = df_out.min(axis=1)
#df_out['max'] = df_out.max(axis=1)
#df_out['std'] = df_out.std(axis=1)

# Take average reading over all sensors for each timestep to yield one target value per timestamp 
df_out = df_out.mean(axis=1)

df_out = (df_out - df_out.min()) / (df_out.max() - df_out.min())
# Normalize each stat column independently (column-wise)
#df_out['mean'] = (df_out['mean'] - df_out['mean'].min()) / (df_out['mean'].max() - df_out['mean'].min())
#df_out['min'] = (df_out['min'] - df_out['min'].min()) / (df_out['min'].max() - df_out['min'].min())
#df_out['max'] = (df_out['max'] - df_out['max'].min()) / (df_out['max'].max() - df_out['max'].min())
#df_out['std'] = df_out['std'] / df_out['std'].max()

# Double the length of df_out
#df_out = df_out.reindex(np.arange(len(df_out) * 2))

# Assign original values to even indices
#df_out.iloc[::2] = df_out.iloc[:len(df_out)//2].values

# Assign 0s to odd indices (nighttime)
#df_out.iloc[1::2] = 0

# Reset index to maintain a clean structure
#df_out.reset_index(drop=True, inplace=True)

#df_out = df_out[:len(df_in)]
#df_in_target_vars = pd.DataFrame({'target_min':df_out['min'],'target_max':df_out['max'],'target_std':df_out['std']})
# Repeat each row 24 times (assuming 24 hourly entries per day)
#df_in_target_vars = df_in_target_vars.loc[df_in_target_vars.index.repeat(24)].reset_index(drop=True)

# Trim in case it's slightly longer than df_in
#df_in_target_vars = df_in_target_vars.iloc[:len(df_in)].copy()

# Concatenate with df_in
#df_in = pd.concat([df_in.reset_index(drop=True), df_in_target_vars], axis=1)

#df_out = df_out['mean'].squeeze()

df_sm_T4N = pd.read_csv('C:/Users/Collin/SUMMA/SUMMA_shared/sapflux/T4n_combine1.csv')
df_sm_T4N = df_sm_T4N.loc[:,['TIMESTAMP','ST_1_Avg','ST_2_Avg','ST_3_Avg']]
df_sm_T4N['TIMESTAMP'] = pd.to_datetime(df_sm_T4N['TIMESTAMP']).dt.tz_localize(None)
df_sm_T4N.set_index('TIMESTAMP',inplace=True)
df_sm = df_sm_T4N.resample('H').mean().reset_index()
df_sm = slice_df_by_date(df_sm, 'TIMESTAMP', 2016, 2, 1, 2019, 9, 30)
df_sm = df_sm[~df_sm['TIMESTAMP'].dt.month.isin([10,11,12,1])]
#df_in = df_in.set_index('timestamp',drop=False,inplace=True)
#df_sm = df_sm.set_index('TIMESTAMP',drop=False,inplace=True)
#df_sm.index.tz_localize(None)
#df_in.index.tz_localize(None)
#df_sm = df_sm.reindex(df_in.index)
#df_sm = df_sm.loc[:,['TIMESTAMP','VWC_1_Avg','VWC_2_Avg','VWC_3_Avg']]
# Align with df_in
df_sm = df_sm.set_index('TIMESTAMP').reindex(df_in['timestamp'].values).reset_index()
df_sm = df_sm.rename(columns={'index': 'timestamp'})


# Define function to categorize periods
def time_period(hour):
    return 'day' if 4 <= hour < 21 else 'night'

# Create new columns for grouping
df_in['period'] = df_in['timestamp'].dt.hour.apply(time_period)
df_in['date'] = df_in['timestamp'].dt.date

# Compute mean and overwrite df_in
df_in = df_in.groupby(['date', 'period']).mean().reset_index()

# Drop unnecessary columns
#df_in = df_in.drop(columns=['timestamp'], errors='ignore')

# First, prepare df_sm in the same way as df_in
df_sm['period'] = df_sm['TIMESTAMP'].dt.hour.apply(time_period)
df_sm['date'] = df_sm['TIMESTAMP'].dt.date

# Average soil moisture values by date and period (day/night)
df_sm_grouped = df_sm.groupby(['date', 'period']).mean().reset_index()

# Aggregate df_in by date and period again
df_in_grouped = df_in.groupby(['date', 'period']).mean().reset_index()

# Now merge df_sm into df_in using a left join to preserve all df_in rows
df_in = pd.merge(df_in_grouped, df_sm_grouped, on=['date', 'period'], how='left')
# Merge or keep separately
#df_in = pd.concat([df_in.reset_index(drop=True), df_sm[['ST_1_Avg', 'ST_2_Avg', 'ST_3_Avg']]], axis=1)

# List of VWC columns to apply the mask logic to
vwc_cols = ['ST_1_Avg', 'ST_2_Avg', 'ST_3_Avg']

for col in vwc_cols:
    mask_col = col + '_mask'

    # Create a boolean mask: True where original value is NaN
    df_in[mask_col] = df_in[col].isna()

    # Set NaNs in original column to 0
    df_in[col].fillna(0.0, inplace=True)
    
# Drop unnecessary columns
df_in = df_in.drop(columns=['date','period'], errors='ignore')

'''

elevations = [1962, 2124, 2541]
df_in_list = []

for i, elev in enumerate(elevations):
    path = f'C:/Users/Collin/SUMMA/SUMMA_shared/summaTestCases_3.0/testCases_data/inputData/fieldData/sagehen/forcing_sagehen_{elev}.csv'
    df = pd.read_csv(path)
    df = slice_df_by_date(df, 'timestamp', 2016, 2, 1, 2019, 9, 30)
    df['VPD'] = calculate_vpd(df['TA'], df['RH'])
    df = df.loc[:, ['timestamp', 'TA', 'ISWR', 'VPD']]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[~df['timestamp'].dt.month.isin([10,11,12,1])]
    df['period'] = df['timestamp'].dt.hour.apply(time_period)
    df['date'] = df['timestamp'].dt.date
    #df = df.groupby(['date', 'period']).mean().reset_index()
    df['elevation_id'] = i  # embed this later
    df_in_list.append(df)
    
def extract_tower_id(fname):
    tower_id = 0
    ID = str(fname[-7:-4])
    if ID == 'T4S':
        tower_id = 1
    else:
        pass
    return tower_id

out_dir = 'C:/Users/Collin/SUMMA/SUMMA_shared/sapflux/DailyTotals_'
out_paths = [out_dir+'922_T4N.csv', out_dir+'922_T4S.csv', out_dir+'1217_T4N.csv', out_dir+'1217_T4S.csv']

df_out_list = []
for path in out_paths:
    df = pd.read_csv(path)
    df = slice_df_by_date(df, 'Date', 2016, 2, 1, 2019, 9, 30)
    tower_id = extract_tower_id(path)
    df['tower_ID'] = tower_id
    #df_melt = df.melt(id_vars=['Date'], var_name='sensor_id', value_name='sapflow')
    #df_melt['tower_id'] = tower_id
    df_out_list.append(df)
'''

###################################################################################################
#### the dataframe should now be ready to be converted to PyTorch tensors for use in the model ####
###################################################################################################

# Define some model parameters here to use later
n_features = df_in.shape[1]
n_hidden = 16#64#2*n_features**2
n_layers = 1#2
#n_output = df_out.shape[1]
'''
# this is arbitrary for now but select sizes for the training and testing sets
train_set_len = int(0.7*len(df_in))
test_set_len = int(len(df_in) - train_set_len)

# Define 'lookback', for us specifies how to chunk hourly data up (tentatively using 3, 8-hour chunks for 1 24-hr period)
#lookback = 8
#lookback = 24    # Try using 24 to represent assigning 24 hourly input values to 1 daily sum target output 
lookback = 24   # Here, 2 represents one day(04:00-21:00) or night(21:00-04:00) cycle

# use function above to create PyTorch tensors
#X_train, y_train = create_torch_set(df_in[:train_set_len*lookback],df_out[:train_set_len],lookback)
#X_test, y_test = create_torch_set(df_in[train_set_len*lookback:],df_out[train_set_len:],lookback)

# use function above to create PyTorch tensors
#X_train, y_train = create_torch_set(df_in[:train_set_len],df_out[:train_set_len],lookback)
#X_test, y_test = create_torch_set(df_in[train_set_len:],df_out[train_set_len:],lookback)

# use function above to create PyTorch tensors
X_train, y_train = create_torch_set(df_in[:int(0.7*len(df_in))],df_out[:int(0.7*len(df_out))],lookback)
X_test, y_test = create_torch_set(df_in[:int(0.7*len(df_in))],df_out[:int(0.7*len(df_out))],lookback)

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

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
'''
# Define lookback (24 hourly steps = 1 daily output)
lookback = 2

# Number of samples = number of full daily windows
num_samples = len(df_in) // lookback

# Clip to exact multiple of 24
df_in = df_in.iloc[:num_samples * lookback]
df_out = df_out.iloc[:num_samples]

# Split point based on daily samples
split_idx = int(0.8 * num_samples)

# Split hourly input by sample count (not raw index)
X_train_in = df_in.iloc[:split_idx * lookback]
X_test_in = df_in.iloc[split_idx * lookback:]

# Split daily output
y_train_out = df_out.iloc[:split_idx]
y_test_out = df_out.iloc[split_idx:]

# Create torch datasets
X_train, y_train = create_torch_set(X_train_in, y_train_out, lookback)
X_test, y_test = create_torch_set(X_test_in, y_test_out, lookback)

# Move to device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Print shapes
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)

# Define model architecture, for now very simple using 1 LSTM block and 1 linear block 
class Model(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,
                            num_layers=num_layers,batch_first=True,dropout=0.5)#,
                            #bidirectional=True)    # add bidirectionalality due to sparse target values
        self.fc1 = nn.Linear(hidden_size,hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2,1)
        self.relu = nn.ReLU()
        #self.linear = nn.Linear(hidden_size,1)    # need to 2x hidden size for bidirectionality
    def forward(self,x):
        x,_ = self.lstm(x)
        x = self.fc1(x[:,-1,:])
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Call model using specified parameters, name optimizer and loss functions, and create dataLoader
model = Model(n_features,n_hidden,n_layers).float()
model.to(device)
opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)   # set lower learning rate 
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train),shuffle=False,batch_size=1)    # low batch size for comp. efficiency

# specifiy number of epochs to use for training
n_epochs = 1500

# Train model and calculate loss, print for every 100th epoch to monitor training progress
for epoch in range(n_epochs): 
    model.train()
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
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
        #y_pred_train = model(X_train)
        y_pred_train = model(X_train.to(device))
        #train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        #train_rmse = torch.sqrt(masked_mse_loss(y_pred, y_train))
        train_rmse = torch.sqrt(masked_mse_loss(y_pred_train,y_train.to(device)))
        #y_pred = model(X_test)
        y_pred_test = model(X_test.to(device))
        #test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        #test_rmse = torch.sqrt(masked_mse_loss(y_pred, y_test))
        test_rmse = torch.sqrt(masked_mse_loss(y_pred_test,y_test.to(device)))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
    
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns

# Move test tensors to CPU and convert to numpy
y_true = y_test.cpu().numpy().flatten()
y_pred = model(X_test).detach().cpu().numpy().flatten()

# Mask NaNs (common in sapflow data)
mask = ~np.isnan(y_true)
y_true = y_true[mask]
y_pred = y_pred[mask]

# Compute errors and metrics
errors = y_true - y_pred
r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

# Confidence intervals (95%) from residual std
residual_std = np.std(errors)
ci_upper = y_pred + 1.96 * residual_std
ci_lower = y_pred - 1.96 * residual_std

# Plot 1: Time series with 95% CI
plt.figure(figsize=(10, 4))
plt.plot(y_true, label='Actual', linewidth=2)
plt.plot(y_pred, label='Predicted', alpha=0.7)
plt.fill_between(np.arange(len(y_pred)), ci_lower, ci_upper, color='gray', alpha=0.3, label='95% CI')
plt.xlabel('Sample Index')
plt.ylabel('Target Value')
plt.title('Time Series: Actual vs. Predicted with 95% Confidence Interval')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Scatter plot (Prediction vs Actual)
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Scatter: R² = {r2:.2f}, RMSE = {rmse:.2f}')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 3: Residuals
plt.figure(figsize=(10, 4))
plt.plot(errors, label='Residuals')
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Error')
plt.title('Residuals (Actual - Predicted)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 4: Histogram of residuals
plt.figure(figsize=(6, 4))
sns.histplot(errors, bins=30, alpha=0.7)
plt.axvline(0, color='r', linestyle='--')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.grid(True)
plt.tight_layout()
plt.show()

    