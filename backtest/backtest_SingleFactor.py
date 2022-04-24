# -*- coding: utf-8 -*-
"""
Created on Mon Feb 8 22:02:51 2022

@author: Wang
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from factor.factorBacktest import SingleFactorBacktest
rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'
factorInformation = pd.read_pickle(rootPath+'data\\seperateData\\pickleFactors_01_50_gtja191.pickle')
openPriceInfo = pd.read_pickle(rootPath+'data\\pickleMaskingOpenPrice.pickle')

backtestResult = pd.DataFrame()
for factorName in tqdm(list(factorInformation.keys())):
    if'alpha' in factorName:
    # if factorName == 'alpha1':
        # 在计算多头策略时要求factorExposure和openPrice是相同大小的表格
        factorExposure = pd.DataFrame(factorInformation[factorName])
        openPrice = pd.DataFrame(openPriceInfo['openPrice'])
        # 这里要求tradeDateList和openPrice的index的长度是一致的,格式为timestamp
        tradeDateList = openPrice.index.tolist()
        backtest = SingleFactorBacktest(factorName, factorExposure, openPrice, tradeDateList, 'open')
        backtest.analyze()
        backtestResult[factorName] = backtest.performance

backtestResult.to_csv(rootPath+'\\factor\\0150因子回测结果0311.csv',encoding = 'utf_8_sig')