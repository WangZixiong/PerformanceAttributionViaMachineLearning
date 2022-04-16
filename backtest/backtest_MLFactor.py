# -*- coding: utf-8 -*-
"""
Created on Wen Feb 16 15:02:51 2022

@author: Wang
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from factor.factorBacktest import SingleFactorBacktest
rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'
# 线性模型因子读取方式
factorExposure = pd.read_pickle(rootPath+'factor\\LGBM_stk_loading.pickle')
factorExposure = factorExposure['stk_loading']
# factorExposure.drop(['index'],axis = 1,inplace = True)

# 树模型因子读取方式
# factorExposure = pd.read_pickle(rootPath+'factor\\RF_stk_loading.pickle')
# factorExposure = factorExposure['stk_loading']
# factorExposure = pd.read_csv(rootPath+'factor\\LGBM因子载荷矩阵0317.csv',index_col = 0)

# 神经网络因子读取方式
# factorExposure = pd.read_pickle(rootPath+'factor\\prediction\\KNN5Factor.pkl')
# factorExposure = pd.DataFrame(factorExposure)

#### 中性化处理
medianFactorExposure = pd.DataFrame(columns = factorExposure.columns)
medianArray = np.array(factorExposure.median(axis = 1)).reshape([len(factorExposure),1])
medianMatrix = np.repeat(medianArray,len(factorExposure.columns),axis = 1)
medianizedFactorExposure = pd.DataFrame(np.array(factorExposure) - medianMatrix)


openPriceInfo = pd.read_pickle(rootPath+'data\\pickleMaskingOpenPrice.pickle')
openPrice = pd.DataFrame(openPriceInfo['openPrice'])

factorName = 'XGBoost Factor'
# 这里要求tradeDateList和openPrice的index的长度是一致的,格式为timestamp
tradeDateList = openPrice.index.tolist()
backtest = SingleFactorBacktest(factorName, medianizedFactorExposure, openPrice, tradeDateList, 'open')
backtest.analyze()

backtestResult = pd.DataFrame()
backtestResult.loc[factorName, 'IC_mean'] = round(backtest.IC.mean(),4)
backtestResult.loc[factorName, 'ICIR'] = round(backtest.IC.mean() / backtest.IC.std(),4)
backtestResult.loc[factorName, 'rankIC_mean'] = round(backtest.rankIC.mean(),4)

backtestResult.loc[factorName, 'cumRts'] = backtest.cumRts
backtestResult.loc[factorName, 'annualVol'] = backtest.annualVol
backtestResult.loc[factorName, 'annualRts'] = backtest.annualRts

backtestResult.loc[factorName, 'longTurnover'] = backtest.longTurnover
backtestResult.loc[factorName, 'maxDrawdown'] = backtest.maxDrawdown
backtestResult.loc[factorName, 'winRate'] = backtest.winRate
backtestResult.loc[factorName, 'SharpeRatio'] = backtest.SharpeRatio

backtestResult.to_csv(rootPath+'backtest\\LGBM合成因子回测结果0409.csv',encoding = 'utf_8_sig')