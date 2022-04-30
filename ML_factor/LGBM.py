"""
@Time    : 2022/3/1 20:32
@Author  : Wang
@File    : LGBM.py
@Software: PyCharm
"""
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm.sklearn import LGBMRegressor
from lightgbm import plot_importance, early_stopping, log_evaluation
from data.basicData import BasicData
import numpy as np
import pandas as pd
import pickle
import joblib
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'
#%%
# 设置模型参数
params = {
    'boosting_type': 'gbdt',  # GBDT算法为基础
    'objective': 'regression',
    'metric': 'mse',
    'n_jobs': -1,
    'random_seed': 1,
}

param_grid = {
    'learning_rate': [0.1, 0.3, 0.6],
    'num_leaves': [16, 32, 64],
    'max_depth': [-1, 3, 5, 8]
}
class LGBM():
    def __init__(self,tradeType):
        self.T = len(BasicData.basicFactor['sharedInformation']['axis1Time'])
        self.N = len(BasicData.basicFactor['sharedInformation']['axis2Stock'])
        self.rolling_step = 10
        self.batch_size = 50
        self.val_proportion = 0.2
        self.tradeType = tradeType
    def get_tradedays(self):
        self.all_date = pd.DataFrame(BasicData.basicFactor['sharedInformation']['axis1Time'], columns=['date'])
        # self.all_date['date'] = (self.all_date.date - 719529) * 86400
        # self.all_date['date'] = pd.to_datetime(self.all_date.date, unit='s')
        self.all_date = [str(d.date()) for d in self.all_date['date']]
        self.all_date = [int(d.replace('-', '')) for d in self.all_date]
        self.all_date.sort()
    def X_preperation(self):
        self.X = []
        stockNum = len(BasicData.basicFactor['sharedInformation']['axis2Stock'])
        factorNum = 0
        for factorInd,key in tqdm(enumerate(BasicData.basicFactor)):
            if key != 'sharedInformation':
                factorNum += 1
                # 0228因为要预测T+2时刻的收益率，因此时间维度提取-2之前的因子值即可
                currFactorArray = np.array(BasicData.basicFactor[key])
                currFactorAllTimeAllStockValue = currFactorArray[:-2, :]
                # 判断因子值是否存在大量缺省
                non_array = currFactorAllTimeAllStockValue[
                    (~np.isnan(currFactorAllTimeAllStockValue)) & (~np.isinf(currFactorAllTimeAllStockValue))]
                if non_array.shape[0] == 0:
                    print('factor {} is empty'.format(key))
                else:
                    # 去除极端值
                    currFactorAllTimeAllStockValue[np.isinf(currFactorAllTimeAllStockValue)] = 0
                for ind in range(np.shape(currFactorAllTimeAllStockValue)[0]):
                    currFactorAllTimeAllStockValue[ind, :] = (currFactorAllTimeAllStockValue[ind, :] - min(currFactorAllTimeAllStockValue[ind, :])) \
                            / (max(currFactorAllTimeAllStockValue[ind, :]) - min(currFactorAllTimeAllStockValue[ind, :]))
                if key == 'alpha1':
                    AllFactorAllStockAllTimeValue = currFactorAllTimeAllStockValue
                else:
                    AllFactorAllStockAllTimeValue = np.hstack((AllFactorAllStockAllTimeValue, currFactorAllTimeAllStockValue))
        for timeInd,time in tqdm(enumerate(range(np.shape(AllFactorAllStockAllTimeValue)[0]))):
            currTimeAllStocksAllFactor = AllFactorAllStockAllTimeValue[timeInd,:]
            currTimeFactorMatrix = np.zeros([stockNum,factorNum])
            for factorInd in range(factorNum):
                currTimeFactorMatrix[:,factorInd] = currTimeAllStocksAllFactor[factorInd*stockNum:(factorInd+1)*stockNum]
            self.X.append(currTimeFactorMatrix)
            # self.X[timeInd] = currTimeFactorMatrix
        # self.X = AllFactorAllStockAllTimeValue.reshape(np.shape(AllFactorAllStockAllTimeValue)[0], stockNum,
        #                                                int(np.shape(AllFactorAllStockAllTimeValue)[1] / stockNum))
        # self.X 三维数组，[时间，个股数，因子数]
        self.X = np.array(self.X)
        with open(rootPath+r'data\LGBMData\LGBMData_feature2729_1.pkl', 'wb') as file:
            pickle.dump(self.X, file)
    def Y_preperation(self):
        self.get_tradedays()
        self.all_stock = BasicData.basicFactor['sharedInformation']['axis2Stock']
        # 使用本地的maskingOpenPrice格式为[时间，个股数]
        maskingOpenPriceDict = pd.read_pickle(rootPath+r'data\pickleMaskingOpenPrice2729Times.pickle')
        maskingOpenPriceDF = maskingOpenPriceDict['openPrice']
        self.Y = (maskingOpenPriceDF/maskingOpenPriceDF.shift(1))-1
        # 按照当日因子值交易，是按照次日开盘价买入，第三天开盘价卖出，因此因子对应收益率取[2:]，未来可能会出现X与Y对不齐的问题
        self.Y = np.array(self.Y)[2:, :]
        # 未来一周的个股收益率
        # self.Y = (maskingOpenPriceDF.shift(-5) / maskingOpenPriceDF) - 1
        # # 按照当日因子值交易，是按照次日开盘价买入，第三天开盘价卖出，因此因子对应收益率取[2:]，未来可能会出现X与Y对不齐的问题
        # self.Y = np.array(self.Y)[2:np.shape(self.X)[0]+2, :]
        with open(rootPath + r'data\LGBMData\LGBMData_label2729_1.pkl', 'wb') as file:
            pickle.dump(self.Y, file)
    def data_preparation(self):
        print('start data preperation')
        if os.path.exists(rootPath+r'data\LGBMData\LGBMData_feature2729_1.pkl'):
           self.X = pd.read_pickle(rootPath+r'data\LGBMData\LGBMData_feature2729_1.pkl')
        else:
           self.X_preperation()
        if os.path.exists(rootPath + r'data\LGBMData\LGBMData_label2729_1.pkl'):
            self.Y = pd.read_pickle(rootPath + r'data\LGBMData\LGBMData_label2729_1.pkl')
        else:
            self.Y_preperation()
        # 树模型需要非空的输入输出，需要将NaN的样本feature和label填充为0
        np.nan_to_num(self.Y, copy=False)
        np.nan_to_num(self.X, copy=False)

    def train_model(self,train_x, train_y, valid_x, valid_y, params, param_grid):
        g_model = LGBMRegressor(**params)
        gsearch = GridSearchCV(g_model, param_grid=param_grid, scoring='neg_mean_squared_error')
        gsearch.fit(valid_x,valid_y)
        # 更新参数训练模型
        for p in gsearch.best_params_:
            params[p] = gsearch.best_params_[p]
        self.params = params
        model = LGBMRegressor(**params)
        model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], callbacks=[early_stopping(50), log_evaluation(10)])
        return model
    def DefiniteParamsTrainModel(self,train_x, train_y, valid_x, valid_y, params):
        model = LGBMRegressor(**params)
        model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], callbacks=[early_stopping(50), log_evaluation(10)])
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
        for step in tqdm(range(stepRange+1)):
            # step 1 划分训练 验证 测试集
            StartTimeInd,EndTimeInd = step * self.rolling_step,self.batch_size + step * self.rolling_step
            # 0302 按照君遥的lightGBM.py的划分方法，我们把train valid test一起划分出来后按比例再划分
            # 如果想要self.roling_step作为test集长度的话，self.batch_size要够大，满足比例要求
            whole_X = self.X[StartTimeInd:EndTimeInd,:,:]
            whole_X = whole_X.reshape(np.shape(whole_X)[0]*np.shape(whole_X)[1],np.shape(whole_X)[2])
            whole_Y = self.Y[StartTimeInd:EndTimeInd,:]
            whole_Y = whole_Y.reshape(np.shape(whole_Y)[0]*np.shape(whole_Y)[1],1)
            train_X,valid_X,train_Y,valid_Y = train_test_split(whole_X, whole_Y,test_size=0.2)
            # valid_X, test_X, valid_Y, test_Y = train_test_split(valid_X, valid_Y, test_size=self.rolling_step*self.N)
            # step 2 LGBM模型训练和测试
            # model = self.train_model(train_X, train_Y, valid_X, valid_Y, params, param_grid)
            # 为了加快训练速度，我们提出方案2，即第一次进行grid search，之后沿用前面的params进行fit
            if step == 0:
                model = self.train_model(train_X, train_Y, valid_X, valid_Y, params, param_grid)
            else:
                model = self.DefiniteParamsTrainModel(train_X, train_Y, valid_X, valid_Y, self.params)
            # 0317 保存最后一期模型,保存每一期feature importance
            featureImportanceDF.loc[step, :] = model.feature_importances_
            joblib.dump(model, 'lightGBMModel.pkl')
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
            print("step={}, startDate = {},endDate = {}, MSE = {}, ZERO_MSE = {}".format(step,self.all_date[StartTimeInd],
                                        self.all_date[EndTimeInd],mse,zeros_mse))
            # plot_importance(model, max_num_features=20)
            # plt.show()
            # step 4 将预测收益率作为因子值记录下来
            for timeInd in range(testStartInd,testEndInd):
                currX = self.X[timeInd,:,:]
                currY = self.Y[timeInd,:]
                currPredY = model.predict(currX)
                factorExposure.loc[timeInd,:] = currPredY
                mse = metrics.mean_squared_error(currY, currPredY)
                zeros_mse = metrics.mean_squared_error(currPredY, np.zeros(np.shape(currY)))
                R2oosDF.loc[timeInd, 'R2oos'] = 1 - mse / zeros_mse
        return [factorExposure,R2oosDF,featureImportanceDF]