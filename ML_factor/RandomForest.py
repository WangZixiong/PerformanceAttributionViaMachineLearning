"""
@Time    : 2022/1/10 14:32
@Author  : Wang
@File    : RandomForest.py
@Software: PyCharm
"""
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
from data.basicData import BasicData
rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'
class RF():
    def __init__(self):
        super().__init__()
        self.T = len(BasicData.basicFactor['sharedInformation']['axis1Time'])
        self.N = len(BasicData.basicFactor['sharedInformation']['axis2Stock'])
        self.rolling_step = 130
        self.batch_size = 500
        self.val_proportion = 0.2
    def get_tradedays(self):
        self.all_date = pd.DataFrame(BasicData.basicFactor['sharedInformation']['axis1Time'], columns=['date'])
        self.all_date['date'] = (self.all_date.date-719529)*86400
        self.all_date['date'] = pd.to_datetime(self.all_date.date, unit='s')
        self.all_date = [str(d.date()) for d in self.all_date['date']]
        self.all_date = [int(d.replace('-', '')) for d in self.all_date]
        self.all_date.sort()
    def X_preperation(self):
        # 提取单只个股在时间序列上的因子信息矩阵
        self.X = list(range(len(BasicData.basicFactor['sharedInformation']['axis2Stock'])))
        for stockInd,stockName in tqdm(enumerate(BasicData.basicFactor['sharedInformation']['axis2Stock'])):
            for factorInd,key in enumerate(BasicData.basicFactor):
                if key != 'sharedInformation':
                    # 因子值归一化
                    currStockAllTimeCurrFactorValue = BasicData.basicFactor[key]['factorMatrix'][:, stockInd].reshape((self.T, 1))
                    currStockAllTimeCurrFactorValue = (currStockAllTimeCurrFactorValue-min(currStockAllTimeCurrFactorValue))/(max(currStockAllTimeCurrFactorValue)-min(currStockAllTimeCurrFactorValue))
                    if factorInd == 0:
                        currStockAllTimeAllFactorValue = currStockAllTimeCurrFactorValue
                    else:
                        currStockAllTimeAllFactorValue = np.hstack((currStockAllTimeAllFactorValue, currStockAllTimeCurrFactorValue))
            self.X[stockInd] = currStockAllTimeAllFactorValue
        self.X = np.array(self.X)
        with open(rootPath+r'\data\RFData_feature.pkl', 'wb') as file:
            pickle.dump(self.X, file)
    def Y_preperation(self):
        self.get_tradedays()
        self.all_stock = BasicData.basicFactor['sharedInformation']['axis2Stock']
        # 提取单只个股的收益率日序列复权值
        self.Y = list(range(len(BasicData.basicFactor['sharedInformation']['axis2Stock'])))
        for stockInd,stock in tqdm(enumerate(BasicData.basicFactor['sharedInformation']['axis2Stock'])):
            allTimeClosePriceDF = BasicData.basicMkt[BasicData.basicMkt['s_info_windcode'].isin([stock])]
            allTimeClosePriceDF.sort_values(['trade_dt'],inplace = True)
            allTimeClosePriceDF.set_index(allTimeClosePriceDF.trade_dt,inplace = True)
            # mapTimeClosePriceDF是以self.all_date为索引，2210个log收益率数据
            mapTimeClosePriceDF = pd.DataFrame(index = self.all_date,columns = allTimeClosePriceDF.columns)
            # filter把运算时间拉长了一倍，但可以解决因子的axis1Time和个股收盘价时间的对应问题
            mapIndexList = list(filter(lambda x: x in self.all_date and x in allTimeClosePriceDF.trade_dt.tolist(),self.all_date))
            mapTimeClosePriceDF.loc[mapIndexList,:] = allTimeClosePriceDF.loc[mapIndexList,:]
            self.Y[stockInd] = np.log((mapTimeClosePriceDF.s_dq_close*mapTimeClosePriceDF.s_dq_adjfactor).astype(float)).diff().values

        self.Y = np.array(self.Y)
        with open(rootPath+r'\data\RFData_label.pkl', 'wb') as file:
            pickle.dump(self.Y, file)

    def data_preparation(self):
        if os.path.exists(rootPath+r'\data\RFData_feature.pkl'):
           self.X = pd.read_pickle(rootPath+r'\data\RFData_feature.pkl')
        else:
           self.X_preperation()
        if os.path.exists(rootPath + r'\data\RFData_label.pkl'):
            self.Y = pd.read_pickle(rootPath + r'\data\RFData_label.pkl')
        else:
            self.Y_preperation()

    def train_test_split_function(self,trainStartDateInd,trainEndDateInd,testStartDateInd,testEndDateInd):
        # 切片取数，整理为一维的输入输出
        X_raw_train = self.X[:,trainStartDateInd:trainEndDateInd,:]
        X_train = X_raw_train.reshape([np.shape(X_raw_train)[0] * np.shape(X_raw_train)[1], np.shape(X_raw_train)[2]])
        y_train = self.Y[:,trainStartDateInd:trainEndDateInd].flatten()

        X_raw_test = self.X[:, testStartDateInd:testEndDateInd, :]
        X_test = X_raw_test.reshape([np.shape(X_raw_test)[0] * np.shape(X_raw_test)[1], np.shape(X_raw_test)[2]])
        y_test = self.Y[:, testStartDateInd:testEndDateInd].flatten()

        # 对于label为0的样本直接剔除
        available_y_train_Index = np.isnan(y_train)
        y_train = y_train[available_y_train_Index == False]
        X_train = X_train[available_y_train_Index == False]

        # 树模型需要非空的输入输出，需要将NaN的样本feature和label填充为0
        np.nan_to_num(X_train, copy=False)
        np.nan_to_num(y_train, copy=False)

        available_y_test_Index = np.isnan(y_test)
        y_test = y_test[available_y_test_Index == False]
        X_test = X_test[available_y_test_Index == False]
        np.nan_to_num(X_test, copy=False)
        np.nan_to_num(y_test, copy=False)
        return {'X_train':X_train,'y_train':y_train,'X_test':X_test,'y_test':y_test}

    def whole_sample_fit(self,MaxDepth,N_Estimators):
        self.get_tradedays()
        model = RandomForestRegressor(criterion = 'mse',n_estimators=N_Estimators,max_depth = MaxDepth)
        trainStartDateInd, trainEndDateInd = 0 , int(np.shape(self.X)[1]*0.8)
        testStartDateInd, testEndDateInd = int(np.shape(self.X)[1]*0.8),np.shape(self.X)[1]
        train_test_dataDict = self.train_test_split_function(trainStartDateInd, trainEndDateInd, testStartDateInd,testEndDateInd)
        X_train, y_train, X_test, y_test = train_test_dataDict['X_train'], train_test_dataDict['y_train'], \
                                           train_test_dataDict['X_test'], train_test_dataDict['y_test']

        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        mse = metrics.mean_squared_error(y_test, y_predict)

        print("startDate = {},endDate = {}, mse={}".format(self.all_date[trainStartDateInd],self.all_date[testStartDateInd], mse))

    def rolling_fit(self):
        self.get_tradedays()
        model = RandomForestRegressor(criterion = 'mse',max_depth= 4)
        R2oosDF = pd.DataFrame()
        for step in tqdm(range((len(self.all_date)-self.batch_size)//self.rolling_step+1)):
            trainStartDateInd, trainEndDateInd = step*self.rolling_step,self.batch_size+step*self.rolling_step
            testStartDateInd, testEndDateInd = self.batch_size+step * self.rolling_step, min(self.batch_size + (step+1) * self.rolling_step,self.T)

            train_test_dataDict =self.train_test_split_function(trainStartDateInd, trainEndDateInd, testStartDateInd, testEndDateInd)
            X_train,y_train,X_test,y_test = train_test_dataDict['X_train'],train_test_dataDict['y_train'],\
                                            train_test_dataDict['X_test'],train_test_dataDict['y_test']

            model.fit(X_train,y_train)
            y_predict = model.predict(X_test)
            mse = metrics.mean_squared_error(y_test,y_predict)
            R2oos = 1-mse/metrics.mean_squared_error(y_test,np.zeros(np.shape(y_test)[0]))
            print("step={}, startDate = {},endDate = {}, Roos2={}".format(step, self.all_date[trainStartDateInd], self.all_date[testStartDateInd],R2oos))
            R2oosDF.loc[self.all_date[testStartDateInd],self.all_date[testEndDateInd]] = float(format(R2oos,'.2g'))
        return R2oosDF