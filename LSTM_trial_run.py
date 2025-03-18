# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 15:34:20 2025

@author: Collin
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

df = pd.read_csv('C:/Users/Collin/Documents/SUMMA/SUMMA_shared/summaTestCases_3.0/output/Default par, Init cond_/1962/figure07/as_csv/vegImpactsTranspire_ballBerry_timestep.csv')

et = (df['scalarLatHeatTotal'].values.astype(np.float32)/2257000)*3600

plt.plot(et)
plt.gca().invert_yaxis()
plt.show()

train_set_len = int(len(et)*0.5)
test_set_len = len(et) - train_set_len

train_set, test_set = et[:train_set_len], et[train_set_len:]

def create_torch_set(data, lookback): 
    X, y = [],[]
    for i in range(len(data)-lookback):
        feature = data[i:i+lookback]
        target = data[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

lookback = 1
X_train, y_train = create_torch_set(train_set,lookback)
X_test, y_test = create_torch_set(test_set,lookback)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)



class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1,hidden_size=50,num_layers=1,batch_first=True)
        self.linear = nn.Linear(50,1)
    def forward(self,x):
        x,_ = self.lstm(x)
        x = self.linear(x)
        return x
    
input_dim = features.shape[1]
model = Model().float()
opt = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train),shuffle=True,batch_size=5)

n_epochs = 1000
for epoch in range(n_epochs): 
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # Validation
    if epoch % 100 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
    