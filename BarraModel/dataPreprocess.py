# -*- coding: utf-8 -*-
"""
Created on Wen Apr 15 18:02:51 2022

@author: Wang
"""
import sklearn.linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV

import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\data\\'
sheetNameDict = {'九坤-净值':'九坤投资筛选基金','幻方-净值':'幻方投资筛选基金','明汯-净值':'明汯投资筛选基金','启林-净值':'启林投资筛选基金'}
allFundDF = pd.DataFrame()
for sheetName in tqdm(sheetNameDict):
    selectedFundName = sheetNameDict[sheetName]
    rawQilinInvestmentDF = pd.read_excel(rootPath+'股票多头-私募基金数据.xlsx',sheet_name= sheetName,index_col = 0,header = 2)
    # 因为目前的机器学习因子最晚到2020-02-07，所以锁定在此日期之前有净值的私募基金
    QilinInvestmentDF = pd.DataFrame()
    for fund in rawQilinInvestmentDF.columns:
        period = rawQilinInvestmentDF.index.tolist()[:3689]
        if sum(rawQilinInvestmentDF.loc[period,fund].fillna(0)) != 0:
            QilinInvestmentDF.loc[:,fund] = rawQilinInvestmentDF.loc[:,fund]
    # 看一下筛选出来的基金的特征，比如在2020-02-07之前多少天的净值数据
    fundFeatureDF = pd.DataFrame()
    for fund in tqdm(QilinInvestmentDF.columns):
        currFundNAV = QilinInvestmentDF.loc[:,fund]
        currFundNAVDays = 0
        currFundStartDay,currFundEndDay = 0,0
        for time in QilinInvestmentDF.index.tolist()[:3689]:
            if currFundNAV[time] > 0:
                currFundNAVDays += 1
                currFundEndDay = time
                if currFundStartDay == 0:
                    currFundStartDay = time
        fundFeatureDF.loc[fund,'startDay'] = currFundStartDay
        fundFeatureDF.loc[fund, 'endDay'] = currFundEndDay
        wholePeriod = (currFundEndDay-currFundStartDay).days
        if wholePeriod/currFundNAVDays < 2:
            fundFeatureDF.loc[fund,'frequency'] = 'daily'
        elif wholePeriod/currFundNAVDays > 5:
            fundFeatureDF.loc[fund,'frequency'] = 'weekly'
        fundFeatureDF.loc[fund,'DayNumber'] = currFundNAVDays
        fundFeatureDF.loc[fund, 'startEndPeriodLength'] = wholePeriod
    # fundFeatureDF.to_csv(rootPath+selectedFundName+'特征.csv',encoding='utf_8_sig')
    # 挑选出来净值数目大于50天的基金作为最终归因的基金
    newQilinInvestmentDF = pd.DataFrame()
    for fund in QilinInvestmentDF.columns:
        if fundFeatureDF.loc[fund,'DayNumber'] > 50:
            newQilinInvestmentDF.loc[:,fund] = QilinInvestmentDF.loc[:,fund]
    # newQilinInvestmentDF.to_csv(rootPath+selectedFundName+'.csv',encoding='utf_8_sig')

    # 挑选出来净值数目大于50天，且判断为日频净值数据的基金
    for fund in fundFeatureDF.index:
        if fundFeatureDF.loc[fund,'frequency'] == 'daily' and fundFeatureDF.loc[fund,'DayNumber'] > 50:
            allFundDF.loc[:,fund] = QilinInvestmentDF.loc[:,fund]
# allFundDF.to_csv(rootPath+'股票多头-私募基金筛选-日频.csv',encoding = 'utf_8_sig')