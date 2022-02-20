import json
import pandas as pd
import numpy as np
import keras
import time
import tensorflow as tf

import torch
from torch import nn
from torch import optim
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Tool 1: Calculate flops


def get_flops(model):
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(
        graph=K.get_session().graph, run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops


'''
    Step1:Dimensionality reduction of high-dimensional semantic vectors using PCA-PPA. 
    This step reduces the dimensionality of the 300-dimensional semantic vector into 20-dimensional data.
'''
with open('./data/hdfs_semantic_vec.json') as f:
    # Step1-1 open file
    gdp_list = json.load(f)
    value = list(gdp_list.values())

    # Step1-2 PCA: Dimensionality reduction to 20-dimensional data
    from sklearn.decomposition import PCA
    estimator = PCA(n_components=20)
    pca_result = estimator.fit_transform(value)

    # Step1-3 PPA: De-averaged
    ppa_result = []
    result = pca_result - np.mean(pca_result)
    pca = PCA(n_components=20)
    pca_result = pca.fit_transform(result)
    U = pca.components_
    for i, x in enumerate(result):
        for u in U[0:7]:
            x = x - np.dot(u.transpose(), x) * u
        ppa_result.append(list(x))
    ppa_result = np.array(ppa_result)

'''
    Step2: Read training data.In this process it is necessary to ensure a balance between abnormal and normal samples.
'''


def read_data(path, split=0.7):
    logs_series = pd.read_csv(path)
    logs_series = logs_series.values
    label = logs_series[:, 1]
    logs_data = logs_series[:, 0]
    logs = []
    for i in range(0, len(logs_data)):
        padding = np.zeros((300, 20))
        data = logs_data[i]
        data = [int(n) for n in data.split()]
        for j in range(0, len(data)):
            padding[j] = ppa_result[data[j]]
        padding = list(padding)
        logs.append(padding)
    logs = np.array(logs)
    split_boundary = int(logs.shape[0] * split)
    train_x = logs[: split_boundary]
    valid_x = logs[split_boundary:]
    train_y = label[: split_boundary]
    valid_y = label[split_boundary:]
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 20))
    valid_x = np.reshape(valid_x, (valid_x.shape[0], valid_x.shape[1], 20))
    train_y = keras.utils.np_utils.to_categorical(np.array(train_y))
    valid_y = keras.utils.np_utils.to_categorical(np.array(valid_y))
    return train_x, train_y, valid_x, valid_y


# Residual block
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding='same', dilation=dilation_rate)
        self.conv2 = nn.Conv1d(
            out_channels, 1, 3, padding='same', dilation=dilation_rate)
        self.conv3 = nn.Conv1d(in_channels, out_channels,
                               kernel_size, padding='same')
        self.relu = nn.ReLU()

    def forward(self, x):
        r = self.conv1(x)
        r = self.relu(r)
        r = self.conv2(r)
        if x.shape[1] == self.out_channels:
            shortcut = x
        else:
            shortcut = self.conv3(x)
        o = r + shortcut
        o = self.relu(o)
        return o


class TCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.resblock1 = ResBlock(
            in_channels=20, out_channels=3, kernel_size=3, dilation_rate=1)
        self.resblock2 = ResBlock(
            in_channels=3, out_channels=3, kernel_size=3, dilation_rate=2)
        self.resblock3 = ResBlock(
            in_channels=3, out_channels=3, kernel_size=3, dilation_rate=4)
        self.resblock4 = ResBlock(
            in_channels=3, out_channels=3, kernel_size=3, dilation_rate=8)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class TCNDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.X = torch.moveaxis(self.X, 2, 1)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.length = self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.length


# Path
train_path = './data/log_train.csv'
# Training data and valid data
train_x, train_y, valid_x, valid_y = read_data(train_path)

train_ds = TCNDataset(train_x, train_y)
valid_ds = TCNDataset(valid_x, valid_y)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=True)

model = TCN()
model.to(device)
summary(model, input_size=(64, 20, 300))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

model.train()
for e in range(1, 101):
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f}')

torch.save(model.state_dict(), "model/E-TCN_GAP.pth")
