# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 13:55:12 2022

@author: gong1078899525
"""
import os
import pickle
import pandas as pd
import numpy as np
from data.basicData import BasicData
##BasicData.basicMkt and BasicData.basicFactor
import copy
import torch
import time
class Basicfactor_LR:
    def __init__(self):
        if hasattr(BasicData, 'T'):
            self.ini_info=False
        else:
            self.ini_info=True
        self.T=getattr(BasicData, 'T', None)
        self.N=getattr(BasicData, 'N', None)
        # self.train_days=getattr(self, 'backward_day', None)
        self.train_days=BasicData.backward_day
        self.all_date=getattr(BasicData, 'trade_date', None)
        # self.T=BasicData.T
        # self.N=BasicData.N
        # self.train_days=BasicData.backward_day
        # self.all_date=BasicData.trade_date
        pass
    
    # def get_tradedays(self):
    #     self.all_date = pd.DataFrame(
    #         BasicData.basicFactor['sharedInformation']['axis1Time'], columns=['date'])
    #     self.all_date['date'] = (self.all_date.date-719529)*86400
    #     self.all_date['date'] = pd.to_datetime(self.all_date.date, unit='s')
    #     self.all_date = [str(d.date()) for d in self.all_date['date']]
    #     self.all_date = [int(d.replace('-', '')) for d in self.all_date]
    #     self.all_date.sort()

    def X_prepare(self, nums=100):
        print('X_prepare')
        # self.X = list(range(len(BasicData.basicFactor['sharedInformation']['axis2Stock'])))
        self.X = list(
            range(len(BasicData.basicFactor['sharedInformation']['axis2Stock'][:nums])))
        for n, s in enumerate(BasicData.basicFactor['sharedInformation']['axis2Stock'][:nums]):
            ini_flag = False
            for k, (key, value) in enumerate(BasicData.basicFactor.items()):
                if key != 'sharedInformation':
                    if ini_flag == False:
                        ini_flag = True
                        s_factor = value[:,n].reshape((self.T, 1))
                    else:
                        s_factor = np.hstack(
                            (s_factor, value[:, n].reshape((self.T, 1))))

            s_X = list(range(self.T))
            for d in range(self.T):
                s_X[d] = s_factor[d]  # 不设置回看期
            self.X[n] = s_X
            print(n, end=',')
        # with open('./data/LRData_feature.pkl', 'wb') as file:
        #     pickle.dump(self.X, file)

    def Y_prepare(self, nums=100):
        # self.get_tradedays()
        """
        matlab时间戳 转成 python日期的方法，例：734508
        （734508-719529）* 86400
        """
        # 如何复权？
        # 并表
        # self.all_stock = BasicData.basicFactor['sharedInformation']['axis2Stock']
        self.all_stock = BasicData.basicFactor['sharedInformation']['axis2Stock'][:2]
        self.all_return = pd.DataFrame([s for s in self.all_stock for i in range(len(
            self.all_date))], index=self.all_date*len(self.all_stock), columns=['s_info_windcode'])
        self.all_return.set_index(
            [self.all_return.index, self.all_return.s_info_windcode], inplace=True)

        all_return = BasicData.basicMkt[['s_info_windcode', 'trade_dt']].copy()
        all_return['return'] = np.log(BasicData.basicMkt.s_dq_open)
        all_return.sort_values(['s_info_windcode', 'trade_dt'], inplace=True)
        all_return.loc[:, 'return'] = all_return.groupby('s_info_windcode')[
            'return'].diff().fillna(0)
        all_return.set_index(['trade_dt', 's_info_windcode'], inplace=True)

        self.all_return['return'] = all_return['return']
        self.all_return = self.all_return.droplevel(1)
        Y = all_return.unstack('trade_dt').droplevel(None, axis=1)
        new_Y=pd.DataFrame(index=BasicData.basicFactor['sharedInformation']['axis2Stock'],columns=Y.columns)
        new_Y.loc[Y.index]=Y
        # if nums!='all':
        self.Y = new_Y[self.all_date[:]].iloc[:nums, :].fillna(0)
        # else:
            # self.Y = Y[self.all_date[:]]

        self.Y = np.array(self.Y)
        self.Y[:,:-2]=self.Y[:,2:]
        self.Y[:,-2]=np.nan

    def data_preparation(self, ini, nums):
        # if os.path.exists('./data/LRData_feature.pkl') and ini != True:
        #     self.X = pd.read_pickle('./data/LRData_feature.pkl')
        #     self.Y = pd.read_pickle('./data/LRData_label.pkl')
        # else:
        #     if nums == 'all':
        #         nums = self.N
        #     self.X_prepare(nums=nums)
        #     self.Y_prepare(nums=nums)
        #     print('data prepare finished')
        if nums == 'all':
            nums = self.N
        self.X_prepare(nums=nums)
        self.Y_prepare(nums=nums)
        
    
    def clean_data(self,ini, nums):
        self.data_preparation(ini, nums)
        def prepare(array, loca):
            non_array = array[(~np.isnan(array)) & (~np.isinf(array))]
            if non_array.shape[0] != 0:
                posinf = np.percentile(non_array, 95)
                neginf = np.percentile(non_array, 5)
                process_array = np.nan_to_num(
                    array, nan=0, posinf=posinf, neginf=neginf)
                max_value = process_array.max()
                min_value = process_array.min()
                if ~np.isnan(max_value) and ~np.isnan(min_value):
                    minmax_array = (process_array-min_value) / \
                        (max_value-min_value)
                    # 如果min和max相同，则会出现nan，此时填充
                    minmax_array[np.isnan(minmax_array)] = 0
                    return minmax_array
                else:
                    print('!!!!!!!error prepare!!!!!!!')
                    process_array[np.isnan(process_array)] = 0
                    return process_array
            else:
                # print(
                #     f'!!!!!empty factor located in [{loca[0]},{loca[1]}]!!!!')
                return np.zeros(array.shape)
        self.X = np.array(self.X)
        self.X[np.isinf(self.X)] = 0
        self.Y[np.isinf(self.Y)] = 0
        # self.X = torch.Tensor(np.array(self.X))
        # self.Y = torch.Tensor(np.array(self.Y))
        # x_numpy = self.X.numpy()
        x_slice = self.X.copy()
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[2]):
                x_slice[i, :, j] = prepare(self.X[i, :, j], [i, j])
        if np.isnan(x_slice.max()):
            print('x_train data nan error!')
            return x_slice, 0
        print('x的长度为{},y的长度为{}'.format(x_slice[i, :, j].shape[0],str(self.Y.shape)))
        data_info={'T':BasicData.T,'N':BasicData.N,'trade_date':BasicData.trade_date,
                   'last_step':BasicData.last_step,'cur_period':BasicData.cur_period,
                   'backward_day':BasicData.backward_day,'year':BasicData.year,
                   'begin_index':BasicData.begin_index,'end_index':BasicData.end_index}
        with open('./data/LRData_label_{}.pkl'.format(BasicData.year), 'wb') as file:
            pickle.dump({'Y_data':self.Y,'data_info':data_info}, file)
        with open('./data/LRData_feature_{}.pkl'.format(BasicData.year), 'wb') as file:
            pickle.dump(x_slice, file)
    
    def load_data(self, year,ini=False,nums='all'):
        if os.path.exists('./data/LRData_feature_{}.pkl'.format(year)) and ini != True:
            timeseri=[time.time()]
            self.X = pd.read_pickle('./data/LRData_feature_{}.pkl'.format(year)).astype(np.float32)
            print(self.X.shape)
            timeseri.append(time.time())
            self.Y = pd.read_pickle('./data/LRData_label_{}.pkl'.format(year))
            timeseri.append(time.time())
            data_info=self.Y['data_info']
            self.Y =self.Y['Y_data'].astype(np.float32)
            for key in data_info:
                exec('BasicData.{}=data_info["{}"]'.format(key,key))
            timeseri.append(time.time())
            for n,times in enumerate(timeseri[1:]):
                print('第{}次运行的时间为{}'.format(n,times-timeseri[0]))
        else:
            print('数据开始初始化')
            self.clean_data(ini, nums)
            print('data prepare finished')