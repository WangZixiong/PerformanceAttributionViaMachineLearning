""" 
@Time    : 2022/1/8 15:32
@Author  : Carl
@File    : CNN.py
@Software: PyCharm
"""
import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from data.basicData import BasicData
import torch.optim as opt

class CNNModel(nn.Module):
    def __init__(self, kernel_size=3, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=10,
                out_channels=100,
                kernel_size=kernel_size,
                stride=stride,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=kernel_size
            )
        )

        self.output = nn.Linear(600, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        output = self.output(x)
        return output

class CNN(CNNModel):
    def __init__(self):
        super().__init__()
        self.T = len(BasicData.basicFactor['sharedInformation']['axis1Time'])
        self.N = len(BasicData.basicFactor['sharedInformation']['axis2Stock'])
        self.rolling_step = 60
        self.batch_size = 500
        self.val_proportion = 0.2
        self.lookback_num = 10

    def set_paras(self, **kwargs):
        self.rolling_step = kwargs.get('rolling_step')
        self.batch_size = kwargs.get('batch_size')
        self.val_proportion = kwargs.get('val_proportion')

    def get_tradedays(self):
        self.all_date = pd.DataFrame(BasicData.basicFactor['sharedInformation']['axis1Time'], columns=['date'])
        self.all_date['date'] = (self.all_date.date-719529)*86400
        self.all_date['date'] = pd.to_datetime(self.all_date.date, unit='s')
        self.all_date = [str(d.date()) for d in self.all_date['date']]
        self.all_date = [int(d.replace('-', '')) for d in self.all_date]
        self.all_date.sort()

    def X_prepare(self):
        # self.X = list(range(len(BasicData.basicFactor['sharedInformation']['axis2Stock'])))
        self.X = list(range(len(BasicData.basicFactor['sharedInformation']['axis2Stock'][:2])))
        for n, s in enumerate(BasicData.basicFactor['sharedInformation']['axis2Stock'][:2]):
            for k, (key, value) in enumerate(BasicData.basicFactor.items()):
                if key != 'sharedInformation':
                    if k == 0:
                        s_factor = value['factorMatrix'][:, n].reshape((self.T, 1))
                    else:
                        s_factor = np.hstack((s_factor, value['factorMatrix'][:, n].reshape((self.T, 1))))

            s_X = list(range(self.T - 10))
            for d in range(self.lookback_num, self.T):
                s_X[d-10] = s_factor[d-self.lookback_num: d]
            self.X[n] = s_X
        # self.X = np.array(self.X)
        with open('./data/CNNData_feature.pkl', 'wb') as file:
            pickle.dump(self.X, file)

    def Y_prepare(self):
        self.get_tradedays()
        """
        matlab时间戳 转成 python日期的方法，例：734508
        （734508-719529）* 86400
        """
        # 如何复权？
        # 并表
        # self.all_stock = BasicData.basicFactor['sharedInformation']['axis2Stock']
        self.all_stock = BasicData.basicFactor['sharedInformation']['axis2Stock'][:2]
        self.all_return = pd.DataFrame([s for s in self.all_stock for i in range(len(self.all_date))], index=self.all_date*len(self.all_stock), columns=['s_info_windcode'])
        self.all_return.set_index([self.all_return.index, self.all_return.s_info_windcode], inplace=True)

        all_return = BasicData.basicMkt[['s_info_windcode', 'trade_dt']].copy()
        all_return['return'] = np.log(BasicData.basicMkt.s_dq_close)
        all_return.sort_values(['s_info_windcode', 'trade_dt'], inplace=True)
        all_return.loc[:, 'return'] = all_return.groupby('s_info_windcode')['return'].diff()
        all_return.set_index(['trade_dt', 's_info_windcode'], inplace=True)

        self.all_return['return'] = all_return['return']
        self.all_return = self.all_return.droplevel(1)
        self.Y = list(range(len(self.all_stock)))
        for n, s in enumerate(self.all_stock):
            for d in range(self.lookback_num, self.T):
                self.Y[n] = self.all_return.loc[self.all_return.s_info_windcode == s, 'return'].values[10:]
        self.Y = np.array(self.Y)
        with open('./data/CNNData_label.pkl', 'wb') as file:
            pickle.dump(self.Y, file)

    def data_preparation(self):
        if os.path.exists('./data/CNNData_feature.pkl'):
            self.X = pd.read_pickle('./data/CNNData_feature.pkl')
            self.Y = pd.read_pickle('./data/CNNData_label.pkl')
        else:
            self.X_prepare()
            self.Y_prepare()

    # def mse_loss(self, real, pred):
    #     return torch.mean(torch.pow(real-pred, 2))

    def rolling_fit(self):
        self.X = torch.Tensor(np.array(self.X))
        self.Y = torch.Tensor(np.array(self.Y))

        model = CNNModel()
        optimizer = opt.SGD(model.parameters(), lr=0.01)
        loss_func = nn.MSELoss()
        for step in range((self.Y.shape[1]-self.batch_size)//self.rolling_step+1):
            x_train = self.X[:, self.rolling_step*step:self.batch_size+self.rolling_step*step, :, :].flatten(0, 1)
            y_train = self.Y[:, self.rolling_step*step:self.batch_size+self.rolling_step*step].flatten(0,1)
            dataset = TensorDataset(x_train, y_train)
            loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)

            for epoch in range(50):
                for s, (batch_x, batch_y) in enumerate(loader):
                    predict = model(batch_x)
                    loss = loss_func(batch_y.reshape(len(batch_y), 1), predict)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    print("epoch={}, step={}, loss={}".format(step, epoch, s, loss.data.numpy()))
        return model