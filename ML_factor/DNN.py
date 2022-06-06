"""
@Author: Carl
@Time: 2022/2/3 17:52
@SoftWare: PyCharm
@File: KNN.py
"""
import torch
import itertools
import math
import tqdm
import pickle
import time
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


class KNNModel(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.num_inputs = num_inputs

    def set_base_layers(self, neurons=64, num_fc=1):
        self.fc_0 = nn.Sequential(
            nn.Linear(self.num_inputs, neurons),
            nn.ReLU(),
            nn.Dropout()
        )
        self.fc = nn.Sequential(
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Dropout()
        )
        self.fc_layer = nn.ModuleList([self.fc for _ in range(num_fc)])
        self.output = nn.Linear(neurons, 1)

    def forward(self, x):
        x = F.normalize(x, dim=-2, p=2)
        out = self.fc_0(x)
        for fc in self.fc_layer:
            out = fc(out)
        out = self.output(out)
        return out


class KNN:
    def __init__(self):
        self.step = 10
        self.val_proportion = 0.2
        self.rolling_size = 200 + self.step

    def data_load(self):
        self.X = pd.read_pickle('./data/cleanData/x.pkl')
        self.Y = pd.read_pickle('./data/cleanData/y_prediction.pkl')
        # self.X = self.X[:, :-2, :]
        # self.Y = self.Y[:, 2:]

    def init_model(self, num_inputs, neu_range=[64], fc_range=[1, 2, 3, 4, 5]):
        self.model_list = list()
        para_list = list(itertools.product(neu_range, fc_range))
        for para in para_list:
            model = KNNModel(num_inputs)
            model.set_base_layers(*para)
            self.model_list.append(model)

    def get_rolling_data(self, r):
        if r * self.step + self.rolling_size > self.X.shape[0]:
            X = self.X[:, r * self.step:, :]
            Y = self.Y[:, r * self.step:]
        else:
            X = self.X[:, r * self.step:r * self.step + self.rolling_size, :]
            Y = self.Y[:, r * self.step:r * self.step + self.rolling_size]
        X = torch.Tensor(X)
        X = torch.where(torch.isinf(X), torch.full_like(X, 0), X)
        X = torch.where(torch.isnan(X), torch.full_like(X, 0), X)
        Y = torch.Tensor(Y)
        return X, Y

    def rolling_fit(self):
        self.data_load()
        self.init_model(self.X.shape[2])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i, model in enumerate(self.model_list):
            print(f'model{i}')
            for r in range(math.ceil((self.Y.shape[0]-self.rolling_size-1)/self.step)):
                print(f'\n Rolling step:{r} ')
                X, Y = self.get_rolling_data(r)
                X_test = X[-self.step:, :, :].flatten(0, 1)
                Y_test = Y[-self.step:, :].flatten().unsqueeze(1)
                X = X[:-self.step, :, :].flatten(0, 1)
                Y = Y[:-self.step, :].flatten().unsqueeze(1)
                dataset = TensorDataset(X, Y)
                train_set, val_set = torch.utils.data.random_split(dataset, [round(Y.shape[0]*(1-self.val_proportion)),
                                                                             round(Y.shape[0]*self.val_proportion)])
                train_loader = DataLoader(train_set, shuffle=True, batch_size=400)
                val_loader = DataLoader(val_set, shuffle=True, batch_size=400)
                model.to(device)
                criterion = nn.MSELoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
                best_loss = 1e9
                start_time = time.time()
                val_loss = list()
                tra_loss = list()
                for epoch in range(100):
                # for epoch in range(2):
                    # model training
                    model.train()
                    train_loss = list()
                    for inputs, labels in train_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        train_loss.append(loss.item())

                    # model validation
                    model.eval()
                    valid_loss = list()
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        # valid_loss += loss.item()
                        valid_loss.append(loss.item())

                    print(f'Epoch {epoch} \t\t Training Loss: {np.nanmean(train_loss)} \t\t '
                          f'Validation Loss: {np.nanmean(valid_loss)} \t\t'
                          f'Time: {time.time()-start_time}'
                          )
                    val_loss.append(np.nanmean(valid_loss))
                    tra_loss.append(np.nanmean(train_loss))
                    # set early stop
                    if np.nanmean(valid_loss) < best_loss:
                        best_loss = np.nanmean(valid_loss)
                        es = 0
                    else:
                        es += 1
                        print("Counter {} of 3".format(es))
                        if es > 2:
                            print("Early stopping with best_loss: ", best_loss,
                                  "and val_loss for this epoch: ", np.nanmean(valid_loss))
                            break

                prediction = model(X_test)
                test_loss = criterion(prediction, Y_test)
                print(f' Model:{i} Rolling:{r} Loss on test set:{test_loss}')

                plt.plot()
                if r == 0:
                    predictions = prediction.detach().numpy()
                else:
                    predictions = np.hstack((predictions, prediction.detach().numpy()))

            torch.save(model, f"./data/model/KNN{i+1}.pt")
            with open(f'./data/prediction/KNN{i+1}.pkl', 'wb') as file:
                pickle.dump(predictions, file)
