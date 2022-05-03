# -*- coding: utf-8 -*-
"""
Created on Wen May 2 23:02:51 2022

@author: Wang
"""
from tqdm import tqdm
import pandas as pd
import numpy as np
import openpyxl
import pickle
import warnings
warnings.filterwarnings('ignore')
from strategy.StrategyConstruct import Strategy

rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'
subPath = 'TimeSeriesModel\\FFModelResult\\568fund\\LGBM多空因子滚动回归结果\\'
allFundNAV = pd.read_excel(rootPath+'data\\私募基金净值数据\\私募基金初筛\\私募基金筛选.xlsx',index_col= 0)
wb = openpyxl.load_workbook(rootPath+subPath+'私募基金业绩归因Exposure.xlsx')
allFundExposureDict = {}
for fund in wb.sheetnames:
    allFundExposureDict[fund] = pd.read_excel(rootPath+subPath+'私募基金业绩归因Exposure.xlsx',sheet_name=fund,index_col=0)
startTime,endTime = pd.Timestamp(2018,6,30), pd.Timestamp(2020,12,31)

startSeason, endSeason = ['2018Q3','2021Q4']
seasonList = []
seasonDict = {}
for year in range(2015,2022):
    seasonList.append(str(year) + 'Q1')
    seasonList.append(str(year) + 'Q2')
    seasonList.append(str(year) + 'Q3')
    seasonList.append(str(year) + 'Q4')
for season in seasonList:
    seasonStartTime = pd.Timestamp(int(season[:4]),int(season[-1])*3-2,1)
    if season[-1] == '2' or season[-1] == '3':
        seasonEndTime = pd.Timestamp(int(season[:4]),int(season[-1])*3,30)
    else:
        seasonEndTime = pd.Timestamp(int(season[:4]),int(season[-1])*3,31)
    seasonDict[season] = [seasonStartTime,seasonEndTime]
# 统计一下，每个季度都有多少只基金有足够的净值数据
seasonFundNumDF = pd.DataFrame()
for season in seasonList[1:]:
    seasonStartTime, seasonEndTime = seasonDict[season]
    timeList = [time for time in allFundNAV.index.tolist() if time <= seasonEndTime and time >= seasonStartTime]
    currSeasonStrategy = Strategy(seasonStartTime, seasonEndTime, allFundNAV, allFundExposureDict)
    enoughNAVFundList = currSeasonStrategy.selectedAvailableFund()
    seasonFundNumDF.loc[season,'currSeasonFundNum'] = len(enoughNAVFundList)

startSeasonIndex,endSeasonIndex = seasonList.index(startSeason),seasonList.index(endSeason)
allTime = [time for time in allFundNAV.index if time >= startTime and time <= endTime]

allTimeNAV = pd.DataFrame(columns = ['NAV'])
for seasonInd in tqdm(range(startSeasonIndex,endSeasonIndex+1)):
    currSeason = seasonList[seasonInd]
    seasonStartTime, seasonEndTime = seasonDict[currSeason]
    timeList = [time for time in allFundNAV.index.tolist() if time <= seasonEndTime and time >= seasonStartTime]
    currSeasonStrategy = Strategy(seasonStartTime, seasonEndTime, allFundNAV, allFundExposureDict)
    enoughNAVFundList = currSeasonStrategy.selectedAvailableFund()
    if type(enoughNAVFundList) != str:
        currSeasonAllFundCumNAVDF = currSeasonStrategy.calNextPeriodReturn(enoughNAVFundList)
    if seasonInd == startSeasonIndex:
        allTimeNAV = currSeasonAllFundCumNAVDF
    else:
        allTimeNAV = allTimeNAV.append(currSeasonAllFundCumNAVDF*allTimeNAV.iloc[-1,0])

allTimeHighExposureNAV = pd.DataFrame(columns = ['NAV'])
seasonExposureDict = {}
for seasonInd in tqdm(range(startSeasonIndex,endSeasonIndex+1)):
    currSeason = seasonList[seasonInd]
    seasonStartTime,seasonEndTime = seasonDict[currSeason]
    timeList = [time for time in allFundNAV.index.tolist() if time <= seasonEndTime and time >= seasonStartTime]
    currSeasonStrategy = Strategy(seasonStartTime,seasonEndTime,allFundNAV,allFundExposureDict)
    finalSelectedFundList, FundRollingReturnDF = currSeasonStrategy.selectHighExposureFund()
    seasonExposureDict[currSeason] = FundRollingReturnDF
    if type(finalSelectedFundList) != str:
        currSeasonAllFundCumNAVDF = currSeasonStrategy.calNextPeriodReturn(finalSelectedFundList)
    else:
        # 在2018Q3,2018Q4和2020Q2时，筛选出的基金没有超过3只，部分原因是因为，选出来的基金暴露都是负的
        print(currSeason+'没有筛选出来基金')
        currSeasonAllFundCumNAVDF = pd.DataFrame(np.ones([len(timeList),1]),index = timeList,columns = ['NAV'])
    if seasonInd == startSeasonIndex:
        allTimeHighExposureNAV = currSeasonAllFundCumNAVDF
    else:
        allTimeHighExposureNAV = allTimeHighExposureNAV.append(currSeasonAllFundCumNAVDF*allTimeHighExposureNAV.iloc[-1,0])

allTimeHighExposureNAV.to_excel(rootPath+'strategy\\LGBM因子选基策略累计净值.xlsx')
allTimeNAV.to_excel(rootPath+'strategy\\全部量化私募均值累计净值.xlsx')