"""
@Time    : 2022/1/10 14:32
@Author  : Wang
@File    : RandomForest.py
@Software: PyCharm
"""
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import pandas as pd
import pickle
import os
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
from data.basicData import BasicData
import warnings
warnings.filterwarnings("ignore")
# 设置模型参数
params = {
    # 'criterion': 'squared_error',
    'min_samples_split':2,
    'n_jobs': -1,
    'random_state': 1,
}

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [2, 3, 5]
}
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
                currFactorAllTimeAllStockValue = BasicData.basicFactor[key]['factorMatrix'][:-2, :]
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
                    if factorInd == 1:
                        AllFactorAllStockAllTimeValue = currFactorAllTimeAllStockValue
                    else:
                        AllFactorAllStockAllTimeValue = np.hstack((AllFactorAllStockAllTimeValue, currFactorAllTimeAllStockValue))
                    # AllFactorAllStockAllTimeValue最终是时间*（因子数*个股数）的格式
        for timeInd,time in tqdm(enumerate(range(np.shape(AllFactorAllStockAllTimeValue)[0]))):
            currTimeAllStocksAllFactor = AllFactorAllStockAllTimeValue[timeInd,:]
            currTimeFactorMatrix = np.zeros([stockNum,factorNum])
            for factorInd in range(factorNum):
                currTimeFactorMatrix[:,factorInd] = currTimeAllStocksAllFactor[factorInd*stockNum:(factorInd+1)*stockNum]
            self.X.append(currTimeFactorMatrix)
                # self.X 三维数组，[时间，个股数，因子数]
        self.X = np.array(self.X)
        with open(BasicData.rootPath+r'./data/RFData/RFData_feature.pkl', 'wb') as file:
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
        with open(BasicData.rootPath+r'./data/RFData/RFData_label.pkl', 'wb') as file:
            pickle.dump(self.Y, file)

    def data_preparation(self):
        print('start data preperation')
        if os.path.exists(r'./data/RFData/RFData_label.pkl'):
            self.Y = pd.read_pickle(r'./data/RFData/RFData_label.pkl')
        else:
            self.Y_preperation()
        if os.path.exists(r'./data/RFData/RFData_feature.pkl'):
           self.X = pd.read_pickle(r'./data/RFData/RFData_feature.pkl')
        else:
           self.X_preperation()
        # 树模型需要非空的输入输出，需要将NaN的样本feature和label填充为0
        np.nan_to_num(self.Y, copy=False)
        np.nan_to_num(self.X, copy=False)

    def train_model(self,train_x, train_y, valid_x, valid_y, params, param_grid):
        g_model = RandomForestRegressor(**params)
        gsearch = GridSearchCV(g_model, param_grid=param_grid, scoring='neg_mean_squared_error')
        gsearch.fit(valid_x,valid_y)
        # 更新参数训练模型
        for p in gsearch.best_params_:
            params[p] = gsearch.best_params_[p]
        self.params = params
        model = RandomForestRegressor(**params)
        model.fit(train_x, train_y)
        return model
    def DefiniteParamsTrainModel(self,train_x, train_y, valid_x, valid_y, params):
        model = RandomForestRegressor(**params)
        model.fit(train_x, train_y)
        return model

    def rolling_fit(self):
        self.get_tradedays()
        R2oosDF = pd.DataFrame(index=self.all_date)
        featureImportanceDF = pd.DataFrame(columns=range(np.shape(self.X)[2]))
        # factorExposure是对未来收益率的预测，每行代表一个工作日，每列代表一只个股
        factorExposure = pd.DataFrame(np.zeros([self.T, self.N]))
        if self.tradeType == 'open':
            stepRange = (len(self.all_date) - self.batch_size - 2) // self.rolling_step
        elif self.tradeType == 'close':
            stepRange = (len(self.all_date) - self.batch_size - 1) // self.rolling_step
        print('start rolling_fit loop')
        for step in tqdm(range(stepRange + 1)):
            # step 1 划分训练 验证 测试集
            StartTimeInd, EndTimeInd = step * self.rolling_step, self.batch_size + step * self.rolling_step
            # 0302 按照君遥的lightGBM.py的划分方法，我们把train valid test一起划分出来后按比例再划分
            # 如果想要self.roling_step作为test集长度的话，self.batch_size要够大，满足比例要求
            whole_X = self.X[StartTimeInd:EndTimeInd, :, :]
            whole_X = whole_X.reshape(np.shape(whole_X)[0] * np.shape(whole_X)[1], np.shape(whole_X)[2])
            whole_Y = self.Y[StartTimeInd:EndTimeInd, :]
            whole_Y = whole_Y.reshape(np.shape(whole_Y)[0] * np.shape(whole_Y)[1], 1)
            train_X, valid_X, train_Y, valid_Y = train_test_split(whole_X, whole_Y, test_size=0.2)
            # valid_X, test_X, valid_Y, test_Y = train_test_split(valid_X, valid_Y, test_size=self.rolling_step * self.N)
            # step 2 LGBM模型训练和测试
            # model = self.train_model(train_X, train_Y, valid_X, valid_Y, params, param_grid)
            # 为了加快训练速度，我们提出方案2，即第一次进行grid search，之后沿用前面的params进行fit
            if step == 0:
                model = self.train_model(train_X, train_Y, valid_X, valid_Y, params, param_grid)
            else:
                model = self.DefiniteParamsTrainModel(train_X, train_Y, valid_X, valid_Y, self.params)
            # 0317 保存每一期feature importance,最后一期模型
            joblib.dump(model, 'RFModel.pkl')
            featureImportanceDF.loc[step, :] = model.feature_importances_
            # 0317 画出训练误差曲线
            results = model.evals_result()
            epochs = len(results['validation_0']['rmse'])
            x_axis = range(0, epochs)
            # plot log loss
            fig, ax = pyplot.subplots()
            ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
            ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
            ax.legend()
            pyplot.ylabel('RMSE Loss')
            pyplot.title('RF Loss')
            # pyplot.show()
            # 画出feature importance曲线
            # plot_importance(model, max_num_features=20)
            # plt.show()
            # step 3 展示模型预测效果,均方误差和R2oos误差
            testStartInd, testEndInd = self.batch_size + step * self.rolling_step, min(
                self.batch_size + (step + 1) * self.rolling_step, np.shape(self.X)[0])
            test_X = self.X[testStartInd:testEndInd, :, :]
            test_Y = self.Y[testStartInd:testEndInd, :]
            test_X = test_X.reshape(np.shape(test_X)[0] * np.shape(test_X)[1], np.shape(test_X)[2])
            test_Y = test_Y.reshape(np.shape(test_Y)[0] * np.shape(test_Y)[1], 1)
            pred_Y = model.predict(test_X)
            mse = metrics.mean_squared_error(test_Y, pred_Y)
            zeros_mse = metrics.mean_squared_error(pred_Y, np.zeros(np.shape(pred_Y)))
            print("step={}, startDate = {},endDate = {}, MSE = {}, ZERO_MSE = {}".format(step,
                                                                                         self.all_date[StartTimeInd],
                                                                                         self.all_date[EndTimeInd], mse,
                                                                                         zeros_mse))
            # step 4 将预测收益率作为因子值记录下来
            for timeInd in range(testStartInd, testEndInd):
                currX = self.X[timeInd, :, :]
                currY = self.Y[timeInd, :]
                currPredY = model.predict(currX)
                factorExposure.loc[timeInd, :] = currPredY
                mse = metrics.mean_squared_error(currY, currPredY)
                zeros_mse = metrics.mean_squared_error(currPredY, np.zeros(np.shape(currY)))
                R2oosDF.loc[timeInd, 'R2oos'] = 1 - mse / zeros_mse
        return [factorExposure, R2oosDF, featureImportanceDF]