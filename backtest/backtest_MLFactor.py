# -*- coding: utf-8 -*-
"""
Created on Wen Feb 16 15:02:51 2022

@author: Wang
"""
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from factor.factorBacktest import SingleFactorBacktest
rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'
factorName = 'LGBM Factor'

# 线性模型因子读取方式
factorExposure = pd.read_pickle(rootPath+'factor\\newFactors\\linearModel\\ELAST_stk_loading.pickle')
# factorExposure = pd.read_pickle(rootPath+'factor\\ELAST_stk_loading.pickle')
factorExposure = factorExposure['stk_loading']
factorExposure.drop(['index'],axis = 1,inplace = True)

# 树模型因子读取方式
# factorExposure = pd.read_pickle(rootPath+'factor\\RF_stk_loading.pickle')
# factorExposure = factorExposure['stk_loading']
factorExposure = pd.read_csv(rootPath+'factor\\完整版LGBM因子载荷.csv',index_col = 0)

# 神经网络因子读取方式
# factorExposure = pd.read_pickle(rootPath+'factor\\KNNFactor\\KNN5Factor.pkl')
# factorExposure = pd.DataFrame(factorExposure).T


#### 中性化处理
medianFactorExposure = pd.DataFrame(columns = factorExposure.columns)
medianArray = np.array(factorExposure.median(axis = 1)).reshape([len(factorExposure),1])
medianMatrix = np.repeat(medianArray,len(factorExposure.columns),axis = 1)
medianizedFactorExposure = pd.DataFrame(np.array(factorExposure) - medianMatrix)

openPriceInfo = pd.read_pickle(rootPath+'data\\pickleMaskingOpenPrice2729Times.pickle')
# openPriceInfo = pd.read_pickle(rootPath+'data\\pickleMaskingOpenPrice.pickle')
openPrice = pd.DataFrame(openPriceInfo['openPrice'])

# 这里要求tradeDateList和openPrice的index的长度是一致的,格式为timestamp
tradableDateList = openPrice.index.tolist()
backtest = SingleFactorBacktest(factorName, medianizedFactorExposure, openPrice, tradableDateList, 'open','日频')
backtest.analyze()
backtestResult = backtest.performance
longRts,longShortRts = backtest.longRts,backtest.longShortRts
backtestAnnualResult = backtest.annualPerformance
backtestAnnualResult.loc[0,backtestResult.columns] = backtestResult.iloc[0,:]

longRts.fillna(0,inplace=True)
longShortRts.fillna(0,inplace=True)
longRts.to_excel(rootPath+'backtest\\newFactorReturn\\'+factorName+'日频换仓无费率多头收益率0421.xlsx',encoding = 'utf_8_sig')
longShortRts.to_excel(rootPath+'backtest\\newFactorReturn\\'+factorName+'日频换仓无费率多空收益率0421.xlsx',encoding = 'utf_8_sig')
backtestAnnualResult.to_csv(rootPath+'backtest\\newFactors\\'+factorName+'合成因子日频换仓无费率多头回测结果0421.csv',encoding = 'utf_8_sig')

# 0418 修改线性因子
# factorExposure = pd.read_pickle(rootPath+'factor\\newFactors\\linearModel\\LASSO_stk_loading.pickle')
# factorExposure['stk_loading'] = factorExposure['stk_loading'].iloc[:-1,:]
# factorExposure['r2'] = factorExposure['r2'].iloc[:-1,:]
# factorExposure['axis1Time'] = factorExposure['axis1Time'].tolist()
# with open(rootPath+'factor\\newFactors\\linearModel\\LASSO_stk_loading.pickle','wb') as file:
#     pickle.dump(factorExposure,file)