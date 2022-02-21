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
factorInformation = pd.read_pickle(rootPath+'data\\pickleFactors_40factor_gtja191.pickle')
openPriceInfo = pd.read_pickle(rootPath+'data\\pickleMaskingOpenPrice.pickle')

backtestResult = pd.DataFrame()
for factorName in tqdm(list(factorInformation.keys())):
    if'alpha' in factorName:
    # if factorName == 'alpha1':
        # 在计算多头策略时要求factorExposure和openPrice是相同大小的表格
        factorExposure = pd.DataFrame(factorInformation[factorName]['factorMatrix'])
        openPrice = pd.DataFrame(openPriceInfo['openPrice'])
        # 这里要求tradeDateList和openPrice的index的长度是一致的,格式为timestamp
        tradeDateList = openPrice.index.tolist()
        backtest = SingleFactorBacktest(factorName, factorExposure, openPrice, tradeDateList, 'open')
        backtest.analyze()
        backtestResult.loc[factorName,'IC_mean'] = backtest.IC.mean()
        backtestResult.loc[factorName,'ICIR'] = backtest.IC.mean() / backtest.IC.std()
        backtestResult.loc[factorName,'rankIC_mean'] = backtest.rankIC.mean()

        backtestResult.loc[factorName,'cumRts'] = backtest.cumRts
        backtestResult.loc[factorName, 'annualVol'] = backtest.annualVol
        backtestResult.loc[factorName,'annualRts'] = backtest.annualRts
        backtestResult.loc[factorName, 'longTurnover'] = backtest.longTurnover
        backtestResult.loc[factorName, 'maxDrawdown'] = backtest.maxDrawdown
        backtestResult.loc[factorName, 'winRate'] = backtest.winRate
        backtestResult.loc[factorName, 'SharpeRatio'] = backtest.SharpeRatio

backtestResult.to_csv(rootPath+'\\factor\\40因子回测结果0216.csv',encoding = 'utf_8_sig')