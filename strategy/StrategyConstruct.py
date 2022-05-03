# -*- coding: utf-8 -*-
"""
Created on Wen May 2 18:02:51 2022
本文用于根据机器学习合成的价量因子的因子收益情况构建季频策略
将策略与偏股混合型基金指数进行对比
@author: Wang
"""
from tqdm import tqdm
import pandas as pd
import numpy as np

import pickle
import warnings
warnings.filterwarnings('ignore')
rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'
class Strategy():
    def __init__(self,startTime,endTime,allFundNAV,allFundExposureDict):
        # startTime，endTime格式为timestamp
        self.startTime = startTime
        self.endTime = endTime
        # allFundNAV的columns是基金名称，index是时间
        self.allFundNAV = allFundNAV
        self.allFundExposureDict = allFundExposureDict
    # 在换仓时点，根据当时的基金收益情况，对基金进行排序，挑选价量类因子收益最高，且未来继续存续到下一次换仓的十只基金
    def selectHighExposureFund(self):
        beforeTimeList = [time for time in self.allFundNAV.index if time < self.startTime]
        afterTimeList = [time for time in self.allFundNAV.index if time < self.endTime]
        valuationStartTime, valuationEndTime = beforeTimeList[-1], afterTimeList[-1]
        valuationStartTimeInd = self.allFundNAV.index.tolist().index(valuationStartTime)
        valuationEndTimeInd = self.allFundNAV.index.tolist().index(valuationEndTime)
        # !!!!!!!!!!!!!!!!!!!
        # 筛选条件1，未来一个季度净值数据不能小于8个，即至少要有2个月是有净值的
        enoughNAVFundList = []
        for fund in self.allFundNAV.columns.tolist():
            fundInd = self.allFundNAV.columns.tolist().index(fund)
            # 这里用到了未来信息，即判断未来一个季度是否还有足够的净值
            nextSeasonFundNAV = self.allFundNAV.iloc[valuationStartTimeInd:valuationEndTimeInd,fundInd].dropna()
            # 标准是在下一季度净值数目多余8个，为了防止一些节假日导致净值数目过少，但另一方面也引入了一些不再存续的基金
            if len(nextSeasonFundNAV) >= 8:
                enoughNAVFundList.append(fund)
        # 万一没有选出来基金，则直接返回空集
        if len(enoughNAVFundList) == 0:
            startTimeStr,endTimeStr = str(valuationStartTime.year*10000+valuationStartTime.month*100+valuationStartTime.day),
            str(valuationEndTime.year * 10000 + valuationEndTime.month * 100 + valuationEndTime.day)
            print('在时间'+startTimeStr+'至'+endTimeStr+'内，没有净值数目超过8个的基金')
            return 'error','error'
        # !!!!!!!!!!!!!!!!!!!
        # 筛选条件2，过去一年的基金因子收益率排序
        factorExposureReturnAttributionDF = pd.DataFrame()
        FundRollingReturnDF = pd.DataFrame()
        for fund in enoughNAVFundList:
            if fund in self.allFundExposureDict:
                RollingEndTimeInd = 100*valuationStartTime.year + valuationStartTime.month
                # 如果rolling结束时间为201912，那开始时间为201901；如果结束时间为201906，那开始时间为201807
                if valuationStartTime.month == 12:
                    RollingStartTimeInd = 100*valuationStartTime.year + 1
                else:
                    RollingStartTimeInd = 100*(valuationStartTime.year-1) + valuationStartTime.month+1
                rollingIndex = str(RollingStartTimeInd) + '-'+ str(RollingEndTimeInd)
                if rollingIndex in self.allFundExposureDict[fund].index:
                    currRollingWindowFactorReturn = self.allFundExposureDict[fund].loc[rollingIndex,'factorReturnCumReturnAttribution']
                    factorExposureReturnAttributionDF.loc[fund,'factorReturnCumReturnAttribution'] = currRollingWindowFactorReturn

                    # 如果选出来的这些基金，因子收益率贡献都是负的怎么办呢，我们需要看一眼这些基金的因子收益率贡献了多少
                    FundRollingReturnDF.loc[fund, 'factorReturnCumReturnAttribution'] = currRollingWindowFactorReturn
                    FundRollingReturnDF.loc[fund, 'NAVReturnCumReturn'] = self.allFundExposureDict[fund].loc[rollingIndex,'NAVReturnCumReturn']
        if factorExposureReturnAttributionDF.empty == 0:
            factorExposureReturnAttributionDF.sort_values(by = 'factorReturnCumReturnAttribution',inplace= True,ascending= False)
            # 这里的筛选标准是选前10个，但把因子载荷拿出来看，发现有许多有因子载荷的基金，载荷都是负的
            positiveExposureFundNum = 0
            finalSelectedFundList = []
            for fund in factorExposureReturnAttributionDF.index:
                # if factorExposureReturnAttributionDF.loc[fund,'factorReturnCumReturnAttribution'] >0:
                #     finalSelectedFundList.append(fund)
                #     positiveExposureFundNum += 1
                finalSelectedFundList.append(fund)
                positiveExposureFundNum += 1
            finalFundNum = 10 if positiveExposureFundNum > 10 else positiveExposureFundNum
            finalSelectedFundList = finalSelectedFundList[:finalFundNum]
        else:
            print(rollingIndex+'内没有基金在价量类因子上有因子暴露')
            return 'error','error'
        return finalSelectedFundList, FundRollingReturnDF
    def selectedAvailableFund(self):
        beforeTimeList = [time for time in self.allFundNAV.index if time < self.startTime]
        afterTimeList = [time for time in self.allFundNAV.index if time < self.endTime]
        valuationStartTime, valuationEndTime = beforeTimeList[-1], afterTimeList[-1]
        valuationStartTimeInd = self.allFundNAV.index.tolist().index(valuationStartTime)
        valuationEndTimeInd = self.allFundNAV.index.tolist().index(valuationEndTime)
        # !!!!!!!!!!!!!!!!!!!
        # 筛选条件1，未来一个季度净值数据不能小于8个，即至少要有2个月是有净值的
        enoughNAVFundList = []
        for fund in self.allFundNAV.columns.tolist():
            fundInd = self.allFundNAV.columns.tolist().index(fund)
            # 这里用到了未来信息，即判断未来一个季度是否还有足够的净值
            nextSeasonFundNAV = self.allFundNAV.iloc[valuationStartTimeInd:valuationEndTimeInd, fundInd].dropna()
            # 标准是在下一季度净值数目多余8个，为了防止一些节假日导致净值数目过少，但另一方面也引入了一些不再存续的基金
            if len(nextSeasonFundNAV) >= 8:
                enoughNAVFundList.append(fund)
        # !!!!!!!!!!!!!!!!!!!
        # 筛选条件2，在价量类因子上有风险暴露
        finalSelectedFundList = []
        for fund in enoughNAVFundList:
            if fund in self.allFundExposureDict:
                finalSelectedFundList.append(fund)
        return finalSelectedFundList
    # 根据当下的基金名单构建策略，计算未来一个季度的收益
    def calNextPeriodReturn(self,finalSelectedFundList):
        timeList = [time for time in self.allFundNAV.index if time >= self.startTime and time <= self.endTime]
        # 因为有每只基金的净值，我们在买入时采用等权配置，把所有的净值累加起来即可得到N倍的累计净值收益率序列
        allFundCumNAVDF = pd.DataFrame(np.zeros([len(timeList),1]),index = timeList,columns = ['NAV'])
        for fund in tqdm(finalSelectedFundList):
            currFundNAV = self.allFundNAV.loc[timeList,fund].fillna(method='bfill')
            currFundNAV.fillna(method = 'ffill',inplace = True)
            currFundNAV /= currFundNAV[0]
            for time in timeList:
                allFundCumNAVDF.loc[time,['NAV']] += currFundNAV[time]
        allFundCumNAVDF /= allFundCumNAVDF.iloc[0,0]
        return allFundCumNAVDF