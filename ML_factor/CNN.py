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
from data.basicData import BasicData


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class CNNModel(CNN):
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

    # @staticmethod
    def data_transform(self):
        """
        matlab时间戳 转成 python日期的方法，例：734508
        （734508-719529）* 86400
        """
        self.get_tradedays()

        # 如何复权？
        # 并表
        self.all_stock = BasicData.basicFactor['sharedInformation']['axis2Stock']
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
        for n, s in enumerate(BasicData.basicFactor['sharedInformation']['axis2Stock']):
            # s_return = self.all_return.loc[(self.all_return.s_info_windcode == s) & (self.all_return.index.isin(self.all_date)), 'return'].values
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
            self.Y[n] = self.all_return.loc[self.all_return.s_info_windcode == s, 'return'].values[10:]
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

    def X_prepare(self):
        self.X = list(range(len(BasicData.basicFactor['sharedInformation']['axis2Stock'])))
        for n, s in enumerate(BasicData.basicFactor['sharedInformation']['axis2Stock']):
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
        with open('./data/CNNData_train.pkl', 'wb') as file:
            pickle.dump(self.X, file)

    def Y_prepare(self):
        self.Y = list(range(len(self.all_stock)))
        self.get_tradedays()

        # 如何复权？
        # 并表
        self.all_stock = BasicData.basicFactor['sharedInformation']['axis2Stock']
        self.all_return = pd.DataFrame([s for s in self.all_stock for i in range(len(self.all_date))], index=self.all_date*len(self.all_stock), columns=['s_info_windcode'])
        self.all_return.set_index([self.all_return.index, self.all_return.s_info_windcode], inplace=True)

        all_return = BasicData.basicMkt[['s_info_windcode', 'trade_dt']].copy()
        all_return['return'] = np.log(BasicData.basicMkt.s_dq_close)
        all_return.sort_values(['s_info_windcode', 'trade_dt'], inplace=True)
        all_return.loc[:, 'return'] = all_return.groupby('s_info_windcode')['return'].diff()
        all_return.set_index(['trade_dt', 's_info_windcode'], inplace=True)

        self.all_return['return'] = all_return['return']
        self.all_return = self.all_return.droplevel(1)
        for n, s in enumerate(self.all_stock):
            for d in range(self.lookback_num, self.T):
                self.Y[n] = self.all_return.loc[self.all_return.s_info_windcode == s, 'return'].values[10:]
        self.Y = np.array(self.Y)
        with open('./data/CNNData_train.pkl', 'wb') as file:
            pickle.dump(self.Y, file)

    def data_preparation(self):
        if os.path.exists('./data/CNNData.pkl'):
            self.data_all = pd.read('./data/CNNData.pkl')
        else:
            self.data_transform()































