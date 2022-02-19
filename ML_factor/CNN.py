""" 
@Time    : 2022/1/8 15:32
@Author  : Carl
@File    : CNN.py
@Software: PyCharm
"""
import torch
import itertools
import math
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as opt


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        self.fc_0 = nn.Sequential(
            nn.Linear(64*10*153, 64),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
        )

        # prediction
        self.output = nn.Linear(64, 1)
        # classification
        # self.output = nn.Linear(64, 2)

    def set_base_layers(self, num_conv=1, num_fc=1):
        self.conv_layer = nn.ModuleList([self.conv for _ in range(num_conv)])
        self.fc_layer = nn.ModuleList([self.fc for _ in range(num_fc)])

    def forward(self, x):
        x = F.normalize(x, dim=-2, p=2)
        out = self.conv_0(x)
        for conv in self.conv_layer:
            out = conv(out)
        out = out.view(x.shape[0], -1)
        out = self.fc_0(out)
        for fc in self.fc_layer:
            out = fc(out)
        out = self.output(out)
        return out

class CNN:
    def __init__(self):
        super().__init__()
        self.step = 1
        self.data_size = 201
        self.val_proportion = 0.2
        self.group_info = pd.read_pickle('./data/cnnData/group_info.pkl')
        # prediction
        self.Y = pd.read_pickle('./data/cleanData/y_prediction.pkl')
        # classification
        # self.Y = pd.read_pickle('.data/cleanData/y_classification.pkl')

    def set_paras(self, **kwargs):
        if kwargs.get('step') is not None:
            self.step = kwargs.get('step')
        if kwargs.get('data_size') is not None:
            self.data_size = kwargs.get('data_size')
        if kwargs.get('val_proportion') is not None:
            self.val_proportion = kwargs.get('val_proportion')

    def initiate_models(self, conv_range=4, fc_range=3):
        self.model_list = list()
        para_list = list(itertools.product(range(1, conv_range), range(1, fc_range)))
        for para in para_list:
            model = CNNModel()
            model.set_base_layers(*para)
            self.model_list.append(model)

    def __data_load(self, group):
        data = pd.read_pickle(f'./data/cnnData/{group}_cnnFactors_gtja191_211211')
        return data

    def data_preparation(self, r):
        Y = self.Y[:, r*self.step+10:r*self.step+self.data_size+10]
        if self.group_info['groupInfo'][r*self.step+9] != self.group_info['groupInfo'][r*self.step+self.data_size+9]:
            group1 = self.group_info['groupInfo'][r*self.step+9]
            group2 = self.group_info['groupInfo'][r*self.step+self.data_size+9]
            X_data = self.__data_load(group1)
            X = np.zeros((self.Y.shape[0], self.data_size, X_data.shape[-2], X_data.shape[-1]))
            if group1 == 2011:
                s1 = r*self.step
            else:
                s1 = r*self.step - self.group_info['groupLimit'][group1-1]
            e1 = self.group_info['groupLimit'][group1] - self.group_info['groupLimit'][group1-1]
            X[:, 0:(e1-s1), :, :] = X_data[:, s1:e1, :, :]
            e2 = self.data_size-(e1-s1)
            X_data = self.__data_load(group2)
            X[:, (e1-s1):, :, :] = X_data[:, 0:e2, :, :]
        else:
            group = self.group_info['groupInfo'][r * self.step + 9]
            X_data = self.__data_load(group)
            if group == 2011:
                start = r*self.step
            else:
                start = r*self.step - self.group_info['groupLimit'][group-1]
            X = X_data[:, start:start+self.data_size, :, :]
        return X, Y

    def rolling_fit(self):
        self.initiate_models()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        for r in range(math.ceil((self.Y.shape[1]-10-self.data_size)/self.step)):
        # for r in range(2):
            print(f'\n Rolling step: {r} ')
            X, Y = self.data_preparation(r)
            X = torch.Tensor(X)
            X = torch.where(torch.isinf(X), torch.full_like(X, 0), X)
            Y = torch.Tensor(Y)
            X_test = X[:, -1, :, :].unsqueeze(1)
            Y_test = Y[:, -1].unsqueeze(1)
            X = X[:, :-1, :, :].reshape(X.shape[0]*(X.shape[1]-1), 1, X.shape[2], X.shape[3])
            Y = Y[:, :-1].reshape(Y.shape[0]*(Y.shape[1]-1), 1)
            dataset = TensorDataset(X, Y)
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [round(Y.shape[0]*0.8), round(Y.shape[0]*0.2)])
            train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=50)
            val_loader = DataLoader(dataset=val_dataset, shuffle=True, batch_size=50)
            for i, model in enumerate(self.model_list):
                model.to(device)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                best_loss = 1e9
                criterion = nn.MSELoss()
                # for epoch in range(2):
                for epoch in range(100):
                    model.train()
                    train_loss = list()
                    for inputs, labels in train_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        optimizer.zero_grad()
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
                          f'Validation Loss: {np.nanmean(valid_loss)}')

                    # set early stop
                    if np.nanmean(valid_loss) < best_loss:
                        best_loss = np.nanmean(valid_loss)
                        es = 0
                        # torch.save(model.state_dict(), "model_" + str(fold) + 'weight.pt')
                    else:
                        es += 1
                        print("Counter {} of 3".format(es))
                        if es > 2:
                            print("Early stopping with best_loss: ", best_loss,
                                  "and val_loss for this epoch: ", np.nanmean(valid_loss))
                            continue

                if i == 0:
                    best_model, best_score = model, best_loss
                elif best_score > best_loss:
                    best_model, best_score = model, best_loss

                pred = best_model(X_test.to(device))
                test_loss = criterion(pred, Y_test)
                print(f'Rolling:{r} Model:{i} Loss on test set:{test_loss}')

            if r == 0:
                predictions = pred.T
            else:
                predictions = torch.row_stack((predictions, pred.T))

        return predictions









