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
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet,ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats.mstats import winsorize
from sklearn import preprocessing
import copy


class LinearModel(torch.nn.Module):  # 从Module继承
    # 必须实现以下两个函数
    # 初始化
    def __init__(self):
        super(LinearModel, self).__init__()  # 调用父类的初始化
        self.linear = torch.nn.Linear(40, 1)  # 构造一个对象，包含权重和偏置
        # Linear的参数为，输入的维度（特征数量，不是样本数量）和输出的维度，以及是否有偏置(默认为True)
    # 前馈过程中进行的计算

    def forward(self, x):  # 这里实际上是一个override
        y_pred = self.linear(x)  # 在这里计算w * x + b 线性模型
        return y_pred


class Penal_linearregre(LinearModel):
    def __init__(self, normalize=0):
        super().__init__()
        self.T = len(BasicData.basicFactor['sharedInformation']['axis1Time'])
        self.N = len(BasicData.basicFactor['sharedInformation']['axis2Stock'])
        self.rolling_step = 10
        self.batch_size = 500
        self.val_proportion = 0.2
        self.lookback_num = 10
        self.normalize = normalize

    def set_paras(self, **kwargs):
        self.rolling_step = kwargs.get('rolling_step')
        self.batch_size = kwargs.get('batch_size')
        self.val_proportion = kwargs.get('val_proportion')

    def get_tradedays(self):
        self.all_date = pd.DataFrame(
            BasicData.basicFactor['sharedInformation']['axis1Time'], columns=['date'])
        self.all_date['date'] = (self.all_date.date-719529)*86400
        self.all_date['date'] = pd.to_datetime(self.all_date.date, unit='s')
        self.all_date = [str(d.date()) for d in self.all_date['date']]
        self.all_date = [int(d.replace('-', '')) for d in self.all_date]
        self.all_date.sort()

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
                        s_factor = value['factorMatrix'][:,
                                                         n].reshape((self.T, 1))
                    else:
                        s_factor = np.hstack(
                            (s_factor, value['factorMatrix'][:, n].reshape((self.T, 1))))

            s_X = list(range(self.T - 10))
            for d in range(self.lookback_num, self.T):
                # s_X[d-10] = s_factor[d-self.lookback_num: d]
                s_X[d-10] = s_factor[d]  # 不设置回看期
            self.X[n] = s_X
            print(n, end=',')
        # self.X = np.array(self.X)
        with open('./data/LRData_feature.pkl', 'wb') as file:
            pickle.dump(self.X, file)

    def Y_prepare(self, nums=100):
        self.get_tradedays()
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
        all_return['return'] = np.log(BasicData.basicMkt.s_dq_close)
        all_return.sort_values(['s_info_windcode', 'trade_dt'], inplace=True)
        all_return.loc[:, 'return'] = all_return.groupby('s_info_windcode')[
            'return'].diff()
        all_return.set_index(['trade_dt', 's_info_windcode'], inplace=True)

        self.all_return['return'] = all_return['return']
        self.all_return = self.all_return.droplevel(1)
        Y = all_return.unstack('trade_dt').droplevel(None, axis=1)
        self.Y = Y[self.all_date[self.lookback_num:]].iloc[:nums, :]

        self.Y = np.array(self.Y)
        with open('./data/LRData_label.pkl', 'wb') as file:
            pickle.dump(self.Y, file)

    def data_preparation(self, ini, nums):
        if os.path.exists('./data/LRData_feature.pkl') and ini != True:
            self.X = pd.read_pickle('./data/LRData_feature.pkl')
            self.Y = pd.read_pickle('./data/LRData_label.pkl')
        else:
            if nums == 'all':
                nums = self.N
            self.X_prepare(nums=nums)
            self.Y_prepare(nums=nums)
            print('data prepare finished')

    def cv_hyper_param(self, x_train, y_train):
        def get_validation(x_train,y_train):
            vali_L=int(x_train.shape[0]*0.1)
            train_index=list(range(x_train.shape[0]))
            validation_index=np.random.choice(train_index,vali_L,replace=False)
            train_index_=list(set(train_index)-set(validation_index))
            x_validation=x_train[validation_index,:]
            y_validation=y_train[validation_index,]
            x_train_=x_train[train_index_,:]
            y_train_=y_train[train_index_]
            return x_train_,y_train_,x_validation,y_validation
        def train_enet(x_train_,y_train_,x_validation,y_validation,i,get_validation):
            try:
                enetcv=ElasticNetCV(l1_ratio=i)
                enetcv=ElasticNetCV()
                enetcv.fit(x_train_, y_train_)
                alpha=enetcv.alpha_
                predict_y=enetcv.predict(x_validation)
                ic = np.corrcoef(predict_y, y_validation)[1, 0]
                return ic,alpha
                # cv_result[ic]=[i,alpha]
            except UnboundLocalError:
                x_train=np.concatenate([x_train_,x_validation],axis=0)
                y_train=np.concatenate([y_train_,y_validation],axis=0)
                x_train_,y_train_,x_validation,y_validation=get_validation(x_train,y_train)
                ic,alpha=train_enet(x_train_,y_train_,x_validation,y_validation,i,get_validation)
                return ic,alpha
        if self.normalize == 1:
            lassocv = LassoCV()
            lassocv.fit(x_train, y_train)
            return lassocv.alpha_
        elif self.normalize == 2:
            ridgecv = RidgeCV()
            ridgecv.fit(x_train, y_train)
            return ridgecv.alpha_
        elif self.normalize == 0:
            return 0
        else:
            # cv_result={}
            # x_train_,y_train_,x_validation,y_validation=get_validation(x_train,y_train)
            # for i in np.power(np.e,[0,-0.5,-1,-2,-5,-10,-20,-30,-100]):
            #     print(i)
            #     ic,alpha=train_enet(x_train_,y_train_,x_validation,y_validation,i,get_validation)
            #     cv_result[ic]={'l1':i,'alpha':alpha}
            #     if i==1:
            #         max_ic=ic
            #     else:
            #         if ic>max_ic:
            #             max_ic=ic
            # return cv_result[max_ic]
            return {'l1':0.5,'alpha':0.5}

                

    def rolling_fit(self):
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
                print(
                    f'!!!!!empty factor located in [{loca[0]},{loca[1]}]!!!!')
                return np.zeros(array.shape)
        self.X = np.array(self.X)
        self.X[np.isinf(self.X)] = 0
        self.Y[np.isinf(self.Y)] = 0
        self.X = torch.Tensor(np.array(self.X))
        self.Y = torch.Tensor(np.array(self.Y))
        x_numpy = self.X.numpy()
        models_info = copy.deepcopy(self.__dict__)
        for keys in ['X', 'Y']:
            res = models_info.pop(keys)
        model_result = {'model_info': models_info}
        x_slice = x_numpy.copy()
        for i in range(x_numpy.shape[0]):
            for j in range(x_numpy.shape[2]):
                x_slice[i, :, j] = prepare(x_numpy[i, :, j], [i, j])
        if np.isnan(x_slice.max()):
            print('x_train data nan error!')
            return x_slice, 0
        x_slice = torch.Tensor(x_slice)
        predict_y = [np.zeros(self.Y[:, 0:self.batch_size].T.shape)]
        for step in range((self.Y.shape[1]-self.batch_size)//self.rolling_step):
            print(step)
            # if step <= 168:
            #     continue
            x_train = x_slice[:, self.rolling_step*step:self.batch_size +
                              self.rolling_step*step, :].flatten(0, 1)
            y_train = self.Y[:, self.rolling_step *
                             step:self.batch_size+self.rolling_step*step].flatten(0, 1)
            x_predict = x_slice[:, 1+self.batch_size+self.rolling_step *
                                step:1+self.batch_size+self.rolling_step*(step+1), :].flatten(0, 1)

            y_predict = self.Y[:, 1+self.batch_size+self.rolling_step *
                               step:1+self.batch_size+self.rolling_step*(step+1)].flatten(0, 1)
            # 补充nan为0
            y_train = torch.Tensor(np.nan_to_num(y_train.numpy()))
            y_predict = torch.Tensor(np.nan_to_num(y_predict.numpy()))
            if np.isnan(x_train.max()):
                print('x_train data nan error!')
                return x_train, y_predict
            try:
                lambda_ = self.cv_hyper_param(
                    x_train.detach().numpy(), y_train.detach().numpy())
            except ValueError:
                print(x_train.detach().numpy().max())
                return x_train.detach().numpy(), y_train.detach().numpy()
            # 基于最佳的lambda值建模
            if self.normalize == 1:
                lasso = Lasso(alpha=lambda_, normalize=True, max_iter=10000)
            elif self.normalize == 0:
                lasso = Lasso(alpha=0, normalize=True, max_iter=10000)
            elif self.normalize == 2:
                lasso = Ridge(alpha=lambda_, normalize=True, max_iter=10000)
            else:
                lasso=ElasticNet(alpha=lambda_['alpha'],l1_ratio=lambda_['l1'])
            # else:
            #     lasso=Ela
            # 对"类"加以数据实体，执行回归系数的运算
            lasso.fit(x_train.detach().numpy(), y_train.detach().numpy())
            # 返回LASSO回归的系数
            res = lasso.intercept_
            lasso_predict = lasso.predict(x_predict.numpy())
            ic = np.corrcoef(lasso_predict, y_predict.numpy())[1, 0]
            mse = mean_squared_error(y_predict.numpy(), lasso_predict)
            r2 = r2_score(y_predict.numpy(), lasso_predict)
            pred_y=lasso_predict.reshape(int(lasso_predict.shape[0]/x_slice.shape[0]), x_slice.shape[0])
            predict_y.append(pred_y)
            model_result[step] = {'train_period': (self.rolling_step*step, self.batch_size+self.rolling_step*step),
                                  'predict_period': (1+self.batch_size+self.rolling_step * step, 1+self.batch_size+self.rolling_step*(step+1)),
                                  'ic': ic, 'mse': mse, 'r2': r2, 'coef': lasso.coef_, 'inter': res,
                                  "y_predict": pred_y,
                                  }
            print('step={},ic={},mse={},r2={}'.format(step, ic, mse, r2))
        predict_y = np.concatenate(predict_y)
        return model_result, predict_y
