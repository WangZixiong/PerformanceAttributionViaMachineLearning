# -*- coding: utf-8 -*-
"""
Created on Wen Apr 15 18:02:51 2022

@author: Wang
"""
import sklearn.linear_model
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
ridgeparams = {
    'fit_intercept':True,
    'copy_X':True,
    'alpha':0.0001
}
class FundPerformanceAttribute():
    def __init__(self,barra_ML_Return,rawFundNAV,year,fund):
        # 第一件事，需要筛选excel中和barra+ML因子时间段重合的基金
        # 第二件事，需要将因子收益率的日度收益按照私募基金的日期转化为对应时间点之间的收益
        # 第三件事，需要用ridge进行回归

        # rawFundNAV默认格式为Series
        self.allTime = [time for time in barra_ML_Return.index if time in rawFundNAV.index]
        self.barra_ML_Return = barra_ML_Return.loc[self.allTime,:]
        self.rawFundNAV = rawFundNAV[self.allTime]
        self.year = year
        self.fund = fund

    def getAdjustedBarraMLReturn_NAVReturn(self,periodTimeList):
        # 计算在timeList中每段时间内的因子收益率与基金净值收益率并返回
        PeriodNAVReturn = pd.DataFrame(index = periodTimeList[1:],columns=['NAVReturn'])
        PeriodFactorReturn = pd.DataFrame(index = periodTimeList[1:])
        for timeInd in range(1,len(periodTimeList)):
            lastNAVTime,currNAVTime = periodTimeList[timeInd-1],periodTimeList[timeInd]
            periodFactorReturn = self.barra_ML_Return.loc[currNAVTime,:]
            lastNAVTimeInd,currNAVTimeInd = self.allTime.index(lastNAVTime),self.allTime.index(currNAVTime)

            if currNAVTimeInd - lastNAVTimeInd > 1:
            # 拉出从上个净值所在日到这个净值所在日之间的所有因子收益率数据，累乘求期间收益率
                currPeriodFactor = self.barra_ML_Return.iloc[lastNAVTimeInd:currNAVTimeInd,:]
                for factor in currPeriodFactor.columns:
                    periodFactorReturn[factor] = (1+currPeriodFactor.loc[:,factor]).cumprod()[-1]-1
            periodNAVReturn = self.rawFundNAV[currNAVTime]/self.rawFundNAV[lastNAVTime]-1

            PeriodNAVReturn.loc[currNAVTime,PeriodNAVReturn.columns] = periodNAVReturn
            PeriodFactorReturn.loc[currNAVTime,self.barra_ML_Return.columns] = periodFactorReturn
        return PeriodFactorReturn,PeriodNAVReturn

    def getFundFactorExposure(self,currFactorPeriodReturn, currFundNAVPeriodReturn):
        # 用sklearn里面的ridge model拟合
        model = sklearn.linear_model.Ridge(**ridgeparams)
        reg = model.fit(currFactorPeriodReturn, currFundNAVPeriodReturn)
        FundFactorExposure, FundAlpha, score = reg.coef_, reg.intercept_, reg.score(currFactorPeriodReturn,
                                                                                         currFundNAVPeriodReturn)
        # 用statsmodels中的sm.OLS拟合，便于看每个变量的t值
        currFactorPeriodReturn = sm.add_constant(currFactorPeriodReturn)
        model = sm.OLS(np.array(currFundNAVPeriodReturn).astype(float),np.array(currFactorPeriodReturn).astype(float))
        results = model.fit()
        FundFactorExposure, FactorTValue, FactorPValue, score = results.params.astype(float),results.tvalues.astype(float),results.pvalues.astype(float),results.rsquared
        return [FundFactorExposure, FactorTValue, FactorPValue, score, results]
    def getExcessReturn(self):
        # 0423 按照年份取基金净值时间区间
        if self.year != 0:
            currYearTime = [i for i in self.allTime if i.year == self.year]
            currYearLastTimeInd = self.allTime.index(currYearTime[-1])
            untilCurrYearTime = self.allTime[:currYearLastTimeInd]
        else:
            untilCurrYearTime = self.allTime
        # 按照基金净值的时间点将时间分成一段一段的
        fundAvailableTime = []
        for time in untilCurrYearTime:
            if self.rawFundNAV[time] > 0:
                fundAvailableTime.append(time)
        if len(fundAvailableTime) >30:
            startNAVTime, endNAVTime = fundAvailableTime[1], fundAvailableTime[-1]
            PeriodFactorReturn, PeriodNAVReturn = self.getAdjustedBarraMLReturn_NAVReturn(fundAvailableTime)
            PeriodFactorReturn.fillna(0,inplace=True)
            FundFactorExposure, FactorTValue, FactorPValue, score,results = self.getFundFactorExposure(PeriodFactorReturn, PeriodNAVReturn)
            alphaDF = pd.DataFrame(index = fundAvailableTime[1:])
            for time in alphaDF.index:
                timeInd = fundAvailableTime.index(time)-1
                alphaDF.loc[time,'alpha'] = results.resid[timeInd]
            return results,alphaDF
        else:
            print(str(self.year)+'年'+self.fund+'净值数据过少')
            return 0


    def getRollingFundExposure(self,lookbackPeriod):
        # 在本函数中，按照基金净值的时间点将时间分成一段一段的
        fundAvailableTime = []
        for time in self.allTime:
            if self.rawFundNAV[time] > 0:
                fundAvailableTime.append(time)
        # 设定rolling长度
        if lookbackPeriod == '季度':
            lookbackLength = 90
        elif lookbackPeriod == '半年度':
            lookbackLength = 180
        elif lookbackPeriod == '年度':
            lookbackLength = 360

        startNAVTime,endNAVTime = fundAvailableTime[1],fundAvailableTime[-1]
        if (endNAVTime-startNAVTime).days < lookbackLength+30:
            print('基金净值的时间跨度过短，无法进行业绩归因')
            return 'error','error','error','error'
        else:
            # 记录滚动风险暴露与MLFactor暴露
            AllPeriodFundFactorExposure, AllPeriodFactorTValue, AllPeriodFactorPValue, AllPeriodScore = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
            AllPeriodMLFactorAttributionPercent = pd.DataFrame()

            # 滚动长度由输入参数决定，滚动频率是一个月一滚动
            # 计算滚动次数
            firstRollingEndTime = startNAVTime+pd.Timedelta(days=lookbackLength)
            rollingMonthAmount = endNAVTime.year*12+endNAVTime.month - firstRollingEndTime.year*12-firstRollingEndTime.month + 1
            wholeStartYear, wholeStartMonth, wholeEndYear, wholeEndMonth = startNAVTime.year,startNAVTime.month,endNAVTime.year,endNAVTime.month

            for rollingNum in tqdm(range(rollingMonthAmount)):
                # 如果是第一个周期，则开始时间为第一个净值收益率时间（这里确认一下是净值收益率时间，不是净值时间！！！！！！！！！）
                if rollingNum == 0:
                    startTime = startNAVTime
                    startTimeYear, startTimeMonth = startTime.year,startTime.month
                    periodStartTime, periodStartTimeInd = startTime,0
                    # 如果不是，则寻找开始月份的第一个净值收益率时间
                else:
                    startTimeYearMonth = wholeStartYear*12+wholeStartMonth+rollingNum
                    startTimeYear,startTimeMonth = startTimeYearMonth//12,startTimeYearMonth%12
                    if startTimeMonth == 0:
                        startTimeYear, startTimeMonth = startTimeYear-1,12
                    # 提取本月的所有净值数值不为0的日期
                    startTimeMonthNAVTimeList = [time for time in fundAvailableTime if time.year == startTimeYear and time.month == startTimeMonth]
                    # 如果本月没有不为0的日期，就用最一开始的日期作为开始日期，有点鲁莽但保证能运行出来
                    if startTimeMonthNAVTimeList == []:
                        periodStartTime, periodStartTimeInd = startNAVTime, 0
                    else:
                        startTimeMonthNAVTimeList.sort()
                        periodStartTime,periodStartTimeInd = startTimeMonthNAVTimeList[0],fundAvailableTime.index(startTimeMonthNAVTimeList[0])
                # 如果是最后一个周期，则结束时间为最后一个净值收益率时间
                if rollingNum == rollingMonthAmount - 1:
                    endTime = endNAVTime
                    endTimeYear, endTimeMonth = endTime.year, endTime.month
                    periodEndTime,periodEndTimeInd = endTime,-1

                # 如果不是，则寻找结束月份的最后一个净值收益率时间
                else:
                    endTimeYearMonth = wholeEndYear * 12 + wholeEndMonth + (rollingNum - rollingMonthAmount)
                    endTimeYear, endTimeMonth = endTimeYearMonth // 12, endTimeYearMonth % 12
                    if endTimeMonth == 0:
                        endTimeYear, endTimeMonth = endTimeYear-1, 12
                    endTimeMonthNAVTimeList = [time for time in fundAvailableTime if
                                                 time.year == endTimeYear and time.month == endTimeMonth]
                    # 如果本月没有不为0的日期，就用最后一个日期作为结束日期，有点鲁莽但保证能运行出来
                    if endTimeMonthNAVTimeList == []:
                        periodEndTime, periodEndTimeInd = endNAVTime,len(fundAvailableTime)-1
                    else:
                        endTimeMonthNAVTimeList.sort()
                        periodEndTime,periodEndTimeInd = endTimeMonthNAVTimeList[-1],fundAvailableTime.index(endTimeMonthNAVTimeList[-1])

                    periodTimeList = fundAvailableTime[periodStartTimeInd:periodEndTimeInd+1]
                    PeriodFactorReturn,PeriodNAVReturn = self.getAdjustedBarraMLReturn_NAVReturn(periodTimeList)
                    [FundFactorExposure, FactorTValue, FactorPValue, score, results] = self.getFundFactorExposure(PeriodFactorReturn, PeriodNAVReturn)
                ind = str(startTimeYear*100+startTimeMonth)+'-'+str(endTimeYear*100+endTimeMonth)
                AllPeriodFundFactorExposure.loc[ind,self.barra_ML_Return.columns.tolist()] = FundFactorExposure
                AllPeriodFactorTValue.loc[ind,self.barra_ML_Return.columns.tolist()] = FactorTValue
                AllPeriodFactorPValue.loc[ind,self.barra_ML_Return.columns.tolist()] = FactorPValue
                AllPeriodScore.loc[ind,'score'] = score

            return AllPeriodFundFactorExposure, AllPeriodFactorTValue, AllPeriodFactorPValue, AllPeriodScore