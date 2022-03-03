"""
@Time    : 2022/1/10 14:32
@Author  : Wang
@File    : RandomForest.py
@Software: PyCharm
"""
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
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
        self.rolling_step = 10
        self.batch_size = 100
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
        stockNum = len(BasicData.basicFactor['sharedInformation']['axis2Stock'])
        factorNum = 0
        # self.X = list(range(len(BasicData.basicFactor['sharedInformation']['axis2Stock'])))
        # self.X = list(np.zeros([len(BasicData.basicFactor)-1,1]))
        self.X = []
        # 0228 提取因子值时，需要清洗数据，分别是去除极端值和归一化
        # 归一化是指一个因子在同一时间上不同个股的因子值映射到[0,1]区间中
        for factorInd, key in tqdm(enumerate(BasicData.basicFactor)):
            if key != 'sharedInformation':
                factorNum += 1
                # 0228因为要预测T+2时刻的收益率，因此时间维度提取-2之前的因子值即可
                currFactorAllTimeAllStockValue = BasicData.basicFactor[key][:-2, :]
                # 判断因子值是否存在大量缺省
                non_array = currFactorAllTimeAllStockValue[(~np.isnan(currFactorAllTimeAllStockValue)) & (~np.isinf(currFactorAllTimeAllStockValue))]
                if non_array.shape[0] == 0:
                    print('factor {} is empty'.format(key))
                else:
                    # 去除极端值
                    currFactorAllTimeAllStockValue[np.isinf(currFactorAllTimeAllStockValue)] = 0
                    # 归一化
                    for ind in range(np.shape(currFactorAllTimeAllStockValue)[0]):
                        currFactorAllTimeAllStockValue[ind,:] = (currFactorAllTimeAllStockValue[ind,:]-min(currFactorAllTimeAllStockValue[ind,:]))\
                                                         /(max(currFactorAllTimeAllStockValue[ind,:])-min(currFactorAllTimeAllStockValue[ind,:]))
                    if factorInd == 0:
                        AllFactorAllStockAllTimeValue = currFactorAllTimeAllStockValue
                    else:
                        AllFactorAllStockAllTimeValue = np.hstack((AllFactorAllStockAllTimeValue, currFactorAllTimeAllStockValue))
                    # self.X[factorInd] = currFactorAllTimeAllStockValue
        for timeInd,time in tqdm(enumerate(range(np.shape(AllFactorAllStockAllTimeValue)[0]))):
            currTimeAllStocksAllFactor = AllFactorAllStockAllTimeValue[timeInd,:]
            currTimeFactorMatrix = np.zeros([stockNum,factorNum])
            for factorInd in range(factorNum):
                currTimeFactorMatrix[:,factorInd] = currTimeAllStocksAllFactor[factorInd*stockNum:(factorInd+1)*stockNum]
            self.X.append(currTimeFactorMatrix)
                # self.X 三维数组，[时间，个股数，因子数]
        self.X = np.array(self.X)
        with open(rootPath+r'\data\RFData\RFData_feature.pkl', 'wb') as file:
            pickle.dump(self.X, file)
    def Y_preperation(self,nums = 300):
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
        self.Y = np.array(self.Y).T
        # 按照当日因子值交易，是按照次日开盘价买入，第三天开盘价卖出，因此因子对应收益率取[2:]，未来可能会出现X与Y对不齐的问题
        self.Y = self.Y[2:, :]
        with open(rootPath+r'\data\RFData\RFData_label.pkl', 'wb') as file:
            pickle.dump(self.Y, file)

    def data_preparation(self):
        print('start data preperation')
        if os.path.exists(rootPath + r'\data\RFData\RFData_label.pkl'):
            self.Y = pd.read_pickle(rootPath + r'\data\RFData\RFData_label.pkl')
        else:
            self.Y_preperation()
        if os.path.exists(rootPath+r'\data\RFData\RFData_feature.pkl'):
           self.X = pd.read_pickle(rootPath+r'\data\RFData\RFData_feature.pkl')
        else:
           self.X_preperation()
        # 树模型需要非空的输入输出，需要将NaN的样本feature和label填充为0
        np.nan_to_num(self.Y, copy=False)
        np.nan_to_num(self.X, copy=False)
    def newvalidation(self,MaxDepth,N_Estimators,startDateInd,endDateInd):
        model = RandomForestRegressor(criterion = 'mse',n_estimators=N_Estimators,max_depth = MaxDepth,n_jobs=-1)
        # model = GradientBoostingRegressor(criterion = 'mse',n_estimators=N_Estimators,max_depth = MaxDepth)
        trainStartDateInd, trainEndDateInd = startDateInd, startDateInd+int((endDateInd-startDateInd)*0.8)
        testStartDateInd, testEndDateInd = startDateInd+int((endDateInd-startDateInd)*0.8), endDateInd
        # train_test_dataDict = self.train_test_split_function(trainStartDateInd, trainEndDateInd, testStartDateInd,testEndDateInd)
        # X_train, y_train, X_test, y_test = train_test_dataDict['X_train'], train_test_dataDict['y_train'], \
        #                                    train_test_dataDict['X_test'], train_test_dataDict['y_test']
        X_train = self.X[trainStartDateInd: trainEndDateInd,: , :]
        X_train = X_train.reshape([np.shape(X_train)[0]* np.shape(X_train)[1], np.shape(X_train)[2]])
        y_train = self.Y[trainStartDateInd: trainEndDateInd,: ]
        y_train = y_train.reshape([np.shape(y_train)[0] * np.shape(y_train)[1], 1])
        X_test = self.X[testStartDateInd: testEndDateInd,:,: ]
        X_test = X_test.reshape([np.shape(X_test)[0] * np.shape(X_test)[1], np.shape(X_test)[2]])
        y_test = self.Y[testStartDateInd: testEndDateInd,:]
        y_test = y_test.reshape([np.shape(y_test)[0] * np.shape(y_test)[1], 1])
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        mse = metrics.mean_squared_error(y_test, y_predict)
        zero_mse = metrics.mean_squared_error(y_predict,np.zeros(np.shape(y_predict)))
        return [mse,zero_mse]
    def fit_and_predict(self,step):
        R2oosDF = pd.DataFrame(index=self.all_date,columns = ['R2oos'])
        factorExposure = pd.DataFrame(np.zeros([self.T, self.N]))
        trainStartDateInd, trainEndDateInd = step * self.rolling_step, self.batch_size + step * self.rolling_step
        testStartDateInd, testEndDateInd = self.batch_size + step * self.rolling_step, self.batch_size + (
                    step + 1) * self.rolling_step
        # 初始化模型参数
        minMSEDict = {'maxDepth': 5, 'nEstimators': 200, 'mse': 1}
        # 通过grid search进行validation选择最佳超参数
        # for maxDepth in [2,3,4]:
        #    for nEstimators in [50,100,200]:
        #        currMSE,zeroMSE = self.newvalidation(maxDepth, nEstimators, trainStartDateInd, trainEndDateInd)
        #        print("maxDepth={}, nEstimators = {},currMSE = {},zeroMSE ={}".format(maxDepth, nEstimators, currMSE,zeroMSE))
        #        if currMSE < minMSEDict['mse']:
        #            minMSEDict['maxDepth'],minMSEDict['nEstimators'],minMSEDict['mse'] = maxDepth,nEstimators,currMSE
        print("step={}, startDate = {},endDate = {}, bestMaxDepth = {}, bestNEstimators = {}".format(step,
                    self.all_date[trainStartDateInd],self.all_date[testStartDateInd], minMSEDict['maxDepth'],minMSEDict['nEstimators']))

        # model = GradientBoostingRegressor(criterion='mse', n_estimators=minMSEDict['nEstimators'],
        #                                   max_depth=minMSEDict['maxDepth'])
        model = RandomForestRegressor(criterion='mse', max_depth=minMSEDict['maxDepth'], n_estimators=minMSEDict['nEstimators'],n_jobs=-1)
        # train_test_dataDict = self.train_test_split_function(trainStartDateInd, trainEndDateInd, testStartDateInd,
        #                                                      testEndDateInd)

        X_train = self.X[trainStartDateInd: trainEndDateInd,: , :]
        X_train = X_train.reshape([np.shape(X_train)[0]* np.shape(X_train)[1], np.shape(X_train)[2]])
        y_train = self.Y[trainStartDateInd: trainEndDateInd,: ]
        y_train = y_train.reshape([np.shape(y_train)[0] * np.shape(y_train)[1], 1])
        X_test = self.X[testStartDateInd: testEndDateInd,:,: ]
        X_test = X_test.reshape([np.shape(X_test)[0] * np.shape(X_test)[1], np.shape(X_test)[2]])
        y_test = self.Y[testStartDateInd: testEndDateInd,:]
        y_test = y_test.reshape([np.shape(y_test)[0] * np.shape(y_test)[1], 1])
        print('开始模型拟合')
        model.fit(X_train, y_train)
        print('完成模型拟合')
        # 方法1：全部输入模型中进行预测，对结果按照时间依次划分得到每日的factorExposure
        # 出现问题是，单只个股多日因子载荷相同
        y_predict = model.predict(X_test)
        predict_mse = metrics.mean_squared_error(y_test,y_predict)
        zero_mse = metrics.mean_squared_error(y_predict,np.zeros(np.shape(y_predict)))
        R2oos = 1-predict_mse/zero_mse
        R2 = model.score(X_test,y_test)
        print("step={}, startDate = {},endDate = {}, R2oos={},R2 = {}".format(step, self.all_date[trainStartDateInd],
                                                                      self.all_date[testStartDateInd], R2oos,R2))
        R2oosDF.loc[self.all_date[testStartDateInd:testEndDateInd],'R2oos'] = float(format(R2oos,'.2g'))

        # 因为是每日预测，预测收益率作为因子载荷
        y_predict = y_predict.reshape([int(np.size(y_predict)/self.N),self.N])
        factorExposure.loc[range(testStartDateInd,testEndDateInd),:] = y_predict

        # 方法2：每日数据依次输入模型中进行预测，得到每日的factorExposure
        for dateInd in tqdm(range(testStartDateInd, testEndDateInd)):
            currX_test = self.X[dateInd, :, :]
            curry_predict = model.predict(currX_test)
            factorExposure.loc[dateInd, :] = curry_predict
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
        for step in tqdm(range(stepRange)):
            results = self.fit_and_predict(step)
            factorExposure = factorExposure.append(results[0])
            R2oosDF = R2oosDF.append(results[1])
        return [factorExposure, R2oosDF]