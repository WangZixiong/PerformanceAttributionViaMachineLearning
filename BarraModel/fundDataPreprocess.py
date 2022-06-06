# -*- coding: utf-8 -*-
"""
Created on Wen Apr 15 18:02:51 2022
本文件用于挑选净值数量足够多的基金，对其策略和公司打上标签，组成初选基金池
一共用到了三个文件：股票多头-私募基金数据。xlsx，几家私募净值数据。xlsx，几家私募产品信息。xlsx
@author: Wang
"""

import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\data\\私募基金净值数据\\'
sheetNameDict = {'九坤-净值':'九坤投资筛选基金','幻方-净值':'幻方投资筛选基金','明汯-净值':'明汯投资筛选基金','启林-净值':'启林投资筛选基金'}
allFundDF = pd.DataFrame()
fundFeatureDF = pd.DataFrame()
for sheetName in tqdm(sheetNameDict):
    selectedFundName = sheetNameDict[sheetName]
    rawQilinInvestmentDF = pd.read_excel(rootPath+'股票多头-私募基金数据.xlsx',sheet_name= sheetName,index_col = 0,header = 2)
    # 首先排除没有净值的私募基金
    QilinInvestmentDF = pd.DataFrame()
    for fund in rawQilinInvestmentDF.columns:
        if sum(rawQilinInvestmentDF.loc[:,fund].fillna(0)) != 0:
            QilinInvestmentDF.loc[:,fund] = rawQilinInvestmentDF.loc[:,fund]
    # 看一下筛选出来的基金的净值数据特征，包括起止日期，净值披露频率，披露天数
    # 记录筛选基金的基金公司，基金公司规模，投资策略
    for fund in tqdm(QilinInvestmentDF.columns):
        currFundNAV = QilinInvestmentDF.loc[:,fund]
        currFundNAVDays = 0
        currFundStartDay,currFundEndDay = 0,0
        for time in QilinInvestmentDF.index.tolist():
            if currFundNAV[time] > 0:
                currFundNAVDays += 1
                currFundEndDay = time
                if currFundStartDay == 0:
                    currFundStartDay = time
        fundFeatureDF.loc[fund,'startDay'] = currFundStartDay
        fundFeatureDF.loc[fund, 'endDay'] = currFundEndDay
        wholePeriod = (currFundEndDay-currFundStartDay).days
        # 粗略的判断一下是日频净值还是周频净值
        if wholePeriod/currFundNAVDays < 2:
            fundFeatureDF.loc[fund,'frequency'] = 'daily'
        elif wholePeriod/currFundNAVDays > 5:
            fundFeatureDF.loc[fund,'frequency'] = 'weekly'
        fundFeatureDF.loc[fund,'DayNumber'] = currFundNAVDays
        fundFeatureDF.loc[fund, 'startEndPeriodLength'] = wholePeriod

        fundFeatureDF.loc[fund, 'companyName'] = sheetName[:2]
        fundFeatureDF.loc[fund, 'companyScale'] = '百亿量化私募'
        fundFeatureDF.loc[fund, 'strategy'] = '股票多头'
    # fundFeatureDF.to_csv(rootPath+selectedFundName+'特征.csv',encoding='utf_8_sig')
    # 挑选出来净值数目大于50天，且判断为日频净值数据的基金
    for fund in QilinInvestmentDF.columns:
        if fundFeatureDF.loc[fund,'DayNumber'] > 50:
            allFundDF.loc[:,fund] = QilinInvestmentDF.loc[:,fund]
# 读取第二次获取的私募基金数据
rawFundNAVDF = pd.read_excel(rootPath+'几家私募净值数据.xlsx',index_col = 0)
rawFundFeatureDF = pd.read_excel(rootPath+'几家私募产品信息.xlsx')
FundNAVDF = pd.DataFrame()
# 首先排除没有净值的私募基金
for fund in rawFundNAVDF.columns:
    if sum(rawFundNAVDF.loc[:, fund].fillna(0)) != 0:
        FundNAVDF.loc[:, fund] = rawFundNAVDF.loc[:, fund]
for fund in tqdm(FundNAVDF.columns):
    # 根据 rawFundFeatureDF 的信息提取基金的公司、公司规模、策略等信息
    if fund in rawFundFeatureDF['fund_name']:
        currFundFeatureInd = rawFundFeatureDF['fund_name'].tolist().index(fund)
        fundFeatureDF.loc[fund, 'companyName'] = rawFundFeatureDF.iloc[currFundFeatureInd,0]
        if fundFeatureDF.loc[fund, 'companyName'] in ['天演资本','赫富投资','鸣石投资','进化论资产']:
            fundFeatureDF.loc[fund, 'companyScale'] = '百亿量化私募'
        elif fundFeatureDF.loc[fund, 'companyName'] in ['衍盛资产','上海锐天投资','上海稳博投资']:
            fundFeatureDF.loc[fund, 'companyScale'] = '50亿量化私募'
        else:
            fundFeatureDF.loc[fund, 'companyScale'] = '十亿量化私募'
        fundFeatureDF.loc[fund, 'strategy'] = rawFundFeatureDF.iloc[currFundFeatureInd,2]

    # 提取基金的净值信息
    currFundNAV = FundNAVDF.loc[:, fund]
    currFundNAVDays = 0
    currFundStartDay, currFundEndDay = 0, 0
    for time in FundNAVDF.index.tolist():
        if currFundNAV[time] > 0:
            currFundNAVDays += 1
            currFundEndDay = time
            if currFundStartDay == 0:
                currFundStartDay = time
    fundFeatureDF.loc[fund, 'startDay'] = currFundStartDay
    fundFeatureDF.loc[fund, 'endDay'] = currFundEndDay
    wholePeriod = (currFundEndDay - currFundStartDay).days
    # 粗略的判断一下是日频净值还是周频净值
    if wholePeriod / currFundNAVDays < 2:
        fundFeatureDF.loc[fund, 'frequency'] = 'daily'
    elif wholePeriod / currFundNAVDays > 5:
        fundFeatureDF.loc[fund, 'frequency'] = 'weekly'
    fundFeatureDF.loc[fund, 'DayNumber'] = currFundNAVDays
    fundFeatureDF.loc[fund, 'startEndPeriodLength'] = wholePeriod
    # 挑选出来净值数目大于50天，且判断为日频净值数据的基金
    if fundFeatureDF.loc[fund,'frequency'] == 'daily' and fundFeatureDF.loc[fund,'DayNumber'] > 50:
        allFundDF.loc[:,fund] = FundNAVDF.loc[:,fund]
selectedFundFeatureDF = pd.DataFrame()
for fund in allFundDF.columns:
    selectedFundFeatureDF.loc[fund,fundFeatureDF.columns] = fundFeatureDF.loc[fund,fundFeatureDF.columns]
allFundDF.to_excel(rootPath+'私募基金初筛\\私募基金筛选.xlsx',encoding = 'utf_8_sig')
selectedFundFeatureDF.to_csv(rootPath+'私募基金初筛\\私募基金筛选特征.csv',encoding = 'utf_8_sig')