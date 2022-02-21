"""
@Time    : 2022/1/10 14:32
@Author  : Wang
@File    : RandomForest.py
@Software: PyCharm
"""
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from joblib import Parallel,delayed
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
from data.basicData import BasicData
rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'
class RF():
    def __init__(self,tradeType):
        super().__init__()
        self.T = len(BasicData.basicFactor['sharedInformation']['axis1Time'])
        self.N = len(BasicData.basicFactor['sharedInformation']['axis2Stock'])
        self.rolling_step = 20
        self.batch_size = 60
        self.val_proportion = 0.2
        self.tradeType = tradeType
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
            currStockAllTimeAllFactorValue = np.zeros([self.T - 2, 1])
            for factorInd,key in enumerate(BasicData.basicFactor):
                if key != 'sharedInformation':
                    # 因子值归一化，前提是没有极端值
                    currStockAllTimeCurrFactorValue = BasicData.basicFactor[key]['factorMatrix'][:-2, stockInd].reshape((self.T-2, 1))
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
            allTimeOpenPriceDF = BasicData.basicMkt[BasicData.basicMkt['s_info_windcode'].isin([stock])]
            allTimeOpenPriceDF.sort_values(['trade_dt'],inplace = True)
            allTimeOpenPriceDF.set_index(allTimeOpenPriceDF.trade_dt,inplace = True)
            # mapTimeClosePriceDF是以self.all_date为索引，2210个log收益率数据
            mapTimeOepnPriceDF = pd.DataFrame(index = self.all_date,columns = allTimeOpenPriceDF.columns)
            # filter把运算时间拉长了一倍，但可以解决因子的axis1Time和个股收盘价时间的对应问题
            mapIndexList = list(filter(lambda x: x in self.all_date and x in allTimeOpenPriceDF.trade_dt.tolist(),self.all_date))
            mapTimeOepnPriceDF.loc[mapIndexList,:] = allTimeOpenPriceDF.loc[mapIndexList,:]
            # self.Y[stockInd] = np.log((mapTimeOepnPriceDF.s_dq_open*mapTimeOepnPriceDF.s_dq_adjfactor).astype(float)).diff().values
            currStockOpenPrice = (mapTimeOepnPriceDF.s_dq_open * mapTimeOepnPriceDF.s_dq_adjfactor).astype(float)

            # 按照当日因子值交易，是按照次日开盘价买入，第三天开盘价卖出，因此因子对应收益率取iloc[2:]，未来可能会出现X与Y对不齐的问题
            self.Y[stockInd] = (currStockOpenPrice/currStockOpenPrice.shift(1) -1)

        self.Y = np.array(self.Y)
        with open(rootPath+r'\data\RFData_label.pkl', 'wb') as file:
            pickle.dump(self.Y, file)

    def data_preparation(self):
        print('start data preperation')
        if os.path.exists(rootPath + r'\data\RFData_label.pkl'):
            self.Y = pd.read_pickle(rootPath + r'\data\RFData_label.pkl')
        else:
            self.Y_preperation()
        if os.path.exists(rootPath+r'\data\RFData_feature.pkl'):
           self.X = pd.read_pickle(rootPath+r'\data\RFData_feature.pkl')
        else:
           self.X_preperation()

    def train_test_split_function(self,trainStartDateInd,trainEndDateInd,testStartDateInd,testEndDateInd):
        featureDimension = np.shape(self.X)[2]

        # 切片取数，整理为一维的输出
        X_raw_train = self.X[:,trainStartDateInd:trainEndDateInd,:]
        X_train = X_raw_train.reshape([np.shape(X_raw_train)[0] * np.shape(X_raw_train)[1] , np.shape(X_raw_train)[2] ])
        y_train = self.Y[:,trainStartDateInd:trainEndDateInd]
        y_train = y_train.reshape([np.shape(y_train)[0] * np.shape(y_train)[1] , 1])

        # label为0的样本认为是非正常交易个股，直接剔除
        # 因为y_train是一维的，X_train是featureDimension维的，调整X_train时需要保证X_train维数不变
        available_y_train_Index = np.isnan(y_train)
        y_train = y_train[available_y_train_Index == False]
        X_train = X_train[(available_y_train_Index == False).repeat(np.shape(X_train)[1]).reshape(np.shape(X_train))]
        X_train = X_train.reshape(int(np.size(X_train)/featureDimension),featureDimension)

        # 树模型需要非空的输入输出，需要将NaN的样本feature和label填充为0
        np.nan_to_num(X_train, copy=False)
        np.nan_to_num(y_train, copy=False)

        # 0211 不在本函数中调整X_test，而是在主程序中每天进行预测，这样能保证每日顺序不被打乱
        # 让每日因子变得清晰，可能可以解决每次训练出现的同一天所有个股的因子载荷相同的情况
        # 一个问题是，在validation的时候，不需要计算每天的因子载荷，
        # 现在采取的方法是，先这么弄，跳过validation，如果这样做有效，则在validation上再添加新的代码
        X_test = self.X[:, testStartDateInd:testEndDateInd, :]
        # X_test = X_test.reshape([np.shape(X_raw_test)[0] * np.shape(X_raw_test)[1] , np.shape(X_raw_test)[2]])
        y_test = self.Y[:, testStartDateInd:testEndDateInd]
        # y_test = y_test.reshape([np.shape(y_test)[0] * np.shape(y_test)[1], 1])
        np.nan_to_num(X_test, copy=False)
        np.nan_to_num(y_test, copy=False)
        return {'X_train':X_train,'y_train':y_train,'X_test':X_test,'y_test':y_test}

    def validation(self,MaxDepth,N_Estimators,startDateInd,endDateInd):
        self.get_tradedays()
        # model = RandomForestRegressor(criterion = 'mse',n_estimators=N_Estimators,max_depth = MaxDepth)
        model = GradientBoostingRegressor(criterion = 'mse',n_estimators=N_Estimators,max_depth = MaxDepth)
        trainStartDateInd, trainEndDateInd = startDateInd , startDateInd+int((endDateInd-startDateInd)*0.8)
        testStartDateInd, testEndDateInd = startDateInd+int((endDateInd-startDateInd)*0.8),endDateInd
        train_test_dataDict = self.train_test_split_function(trainStartDateInd, trainEndDateInd, testStartDateInd,testEndDateInd)
        X_train, y_train, X_test, y_test = train_test_dataDict['X_train'], train_test_dataDict['y_train'], \
                                           train_test_dataDict['X_test'], train_test_dataDict['y_test']

        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        mse = metrics.mean_squared_error(y_test, y_predict)

        # print("Train startDate = {},endDate = {}, mse={}".format(self.all_date[startDateInd],self.all_date[endDateInd], mse))
        return mse
    def fit_and_predict(self,step):
        R2oosDF = pd.DataFrame(index=self.all_date)
        factorExposure = pd.DataFrame(np.zeros([self.T, self.N]))
        trainStartDateInd, trainEndDateInd = step * self.rolling_step, self.batch_size + step * self.rolling_step
        testStartDateInd, testEndDateInd = self.batch_size + step * self.rolling_step, self.batch_size + (
                    step + 1) * self.rolling_step
        # 初始化模型参数
        minMSEDict = {'maxDepth': 5, 'nEstimators': 200, 'mse': 1}
        # 通过grid search进行validation选择最佳超参数
        # for maxDepth in [2,3,4]:
        #    for nEstimators in [50,100,200]:
        #        currMSE = self.validation(maxDepth, nEstimators, trainStartDateInd, trainEndDateInd)
        #        print("maxDepth={}, nEstimators = {},currMSE = {}".format(maxDepth, nEstimators, currMSE))
        #        if currMSE < minMSEDict['mse']:
        #            minMSEDict['maxDepth'],minMSEDict['nEstimators'],minMSEDict['mse'] = maxDepth,nEstimators,currMSE

        model = GradientBoostingRegressor(criterion='mse', n_estimators=minMSEDict['nEstimators'],
                                          max_depth=minMSEDict['maxDepth'])
        # model = RandomForestRegressor(criterion='mse', max_depth=minMSEDict['maxDepth'], n_estimators=minMSEDict['nEstimators'])
        train_test_dataDict = self.train_test_split_function(trainStartDateInd, trainEndDateInd, testStartDateInd,
                                                             testEndDateInd)
        X_train, y_train, X_test, y_test = train_test_dataDict['X_train'], train_test_dataDict['y_train'], \
                                           train_test_dataDict['X_test'], train_test_dataDict['y_test']

        model.fit(X_train, y_train)
        # 方法1：全部输入模型中进行预测，对结果按照时间依次划分得到每日的factorExposure
        # 出现问题是，单只个股多日因子载荷相同
        # y_predict = model.predict(X_test)
        # mse = metrics.mean_squared_error(y_test,y_predict)
        # 因为是每日预测，预测收益率作为因子载荷
        # y_predict = y_predict.reshape([int(np.size(y_predict)/self.N),self.N])
        # factorExposure.loc[range(testStartDateInd,testEndDateInd),:] = y_predict
        # 方法2：每日数据依次输入模型中进行预测，得到每日的factorExposure
        for dateInd in tqdm(range(testStartDateInd, testEndDateInd)):
            currX_test = X_test[:, dateInd - testStartDateInd, :].reshape(np.shape(X_test)[0], np.shape(X_test)[2])
            curry_test = y_test[:, dateInd - testStartDateInd]
            curry_predict = model.predict(currX_test)
            mse = metrics.mean_squared_error(curry_test, curry_predict)
            factorExposure.loc[dateInd, :] = curry_predict
            R2oos = 1-mse/metrics.mean_squared_error(y_test,np.zeros(np.shape(y_test)))
            print("step={}, startDate = {},endDate = {}, R2oos={}".format(step, self.all_date[trainStartDateInd], self.all_date[testStartDateInd],R2oos))
            R2oosDF.loc[self.all_date[testStartDateInd:testEndDateInd],'R2oos'] = float(format(R2oos,'.2g'))
        return [factorExposure,R2oosDF]

    def rolling_fit(self):
        self.get_tradedays()
        R2oosDF = pd.DataFrame(index = self.all_date)

        # factorExposure是对未来收益率的预测，每行代表一个工作日，每列代表一只个股
        factorExposure = pd.DataFrame(np.zeros([self.T,self.N]))
        if self.tradeType == 'open':
            stepRange = (len(self.all_date)-self.batch_size-2)//self.rolling_step
        elif self.tradeType == 'close':
            stepRange = (len(self.all_date) - self.batch_size - 1) // self.rolling_step
        print('start rolling_fit loop')
        results = Parallel(n_jobs=2)(delayed(self.fit_and_predict)(step) for step in range(stepRange))
        for step in tqdm(range(stepRange)):
            factorExposure = factorExposure.append(results[step][0])
            R2oosDF = R2oosDF.append(results[step][1])
        return [factorExposure, R2oosDF]