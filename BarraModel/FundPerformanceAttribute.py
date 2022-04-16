# -*- coding: utf-8 -*-
"""
Created on Wen Apr 15 18:02:51 2022

@author: Wang
"""
import sklearn.linear_model
import numpy as np
import pandas as pd
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
    def __init__(self,barra_ML_Return,rawFundNAV):
        # 第一件事，需要筛选excel中和barra+ML因子时间段重合的基金
        # 第二件事，需要将因子收益率的日度收益按照私募基金的日期转化为周度收益
        # 第三件事，需要用ridge进行回归

        # rawFundNAV默认格式为Series
        self.allTime = [time for time in barra_ML_Return.index if time in rawFundNAV.index]
        self.barra_ML_Return = barra_ML_Return.loc[self.allTime,:]
        self.rawFundNAV = rawFundNAV[self.allTime]
    def getAdjustedBarraMLReturn_NAVReturn(self,periodTimeList):
        # 计算每段时间内的因子收益率与基金净值收益率并返回
        PeriodNAVReturn = pd.DataFrame(index = periodTimeList[1:],columns=['NAVReturn'])
        PeriodFactorReturn = pd.DataFrame(index = periodTimeList[1:])
        for timeInd in range(1,len(periodTimeList)):
            lastNAVTime,currNAVTime = periodTimeList[timeInd-1],periodTimeList[timeInd]
            periodFactorReturn = self.barra_ML_Return.loc[currNAVTime,:]
            lastNAVTimeInd,currNAVTimeInd = self.allTime.index(lastNAVTime),self.allTime.index(currNAVTime)

            if currNAVTimeInd - lastNAVTimeInd > 1:
            # 拉出上个净值所在日到这个净值所在日之间的所有因子收益率数据，累乘求期间收益率
                currPeriodFactor = self.barra_ML_Return.iloc[lastNAVTimeInd:currNAVTimeInd,:]
                for factor in currPeriodFactor.columns:
                    periodFactorReturn[factor] = (1+currPeriodFactor.loc[:,factor]).cumprod()[-1]-1
            periodNAVReturn = self.rawFundNAV[currNAVTime]/self.rawFundNAV[lastNAVTime]-1

            PeriodNAVReturn.loc[currNAVTime,PeriodNAVReturn.columns] = periodNAVReturn
            PeriodFactorReturn.loc[currNAVTime,self.barra_ML_Return.columns] = periodFactorReturn
        return PeriodFactorReturn,PeriodNAVReturn

    def getFundFactorExposure(self,currFactorPeriodReturn, currFundNAVPeriodReturn):
        model = sklearn.linear_model.Ridge(**ridgeparams)
        reg = model.fit(currFactorPeriodReturn, currFundNAVPeriodReturn)
        FundFactorExposure, FundAlpha, score = reg.coef_, reg.intercept_, reg.score(currFactorPeriodReturn,
                                                                                         currFundNAVPeriodReturn)
        return [FundFactorExposure, FundAlpha, score]
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

        startNAVTime,endNAVTime = fundAvailableTime[1],fundAvailableTime[-1]
        if (endNAVTime-startNAVTime).days < lookbackLength+30:
            print('基金净值的时间跨度过短，无法进行业绩归因')
            return 'error','error'
        else:
            # 记录滚动风险暴露与MLFactor暴露
            AllPeriodFundFactorExposure, AllPeriodScore = pd.DataFrame(),pd.DataFrame()

            # 滚动长度由输入参数决定，滚动频率是一个月一滚动
            # 计算滚动次数
            firstRollingEndTime = startNAVTime+pd.Timedelta(days=lookbackLength)
            rollingMonthAmount = endNAVTime.year*12+endNAVTime.month - firstRollingEndTime.year*12-firstRollingEndTime.month + 1
            wholeStartYear,wholeStartMonth, wholeEndYear, wholeEndMonth = startNAVTime.year,startNAVTime.month,endNAVTime.year,endNAVTime.month

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
                    startTimeMonthNAVTimeList = [time for time in fundAvailableTime if time.year == startTimeYear and time.month == startTimeMonth]
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
                    if endTimeYearMonth == 0:
                        endTimeYear, endTimeMonth = endTimeYear-1, 12
                    endTimeMonthNAVTimeList = [time for time in fundAvailableTime if
                                                 time.year == endTimeYear and time.month == endTimeMonth]
                    if endTimeMonthNAVTimeList == []:
                        periodEndTime, periodEndTimeInd = endNAVTime,len(fundAvailableTime)-1
                    else:
                        endTimeMonthNAVTimeList.sort()
                        periodEndTime,periodEndTimeInd = endTimeMonthNAVTimeList[-1],fundAvailableTime.index(endTimeMonthNAVTimeList[-1])

                    periodTimeList = fundAvailableTime[periodStartTimeInd:periodEndTimeInd+1]
                    PeriodFactorReturn,PeriodNAVReturn = self.getAdjustedBarraMLReturn_NAVReturn(periodTimeList)
                    [FundFactorExposure, FundAlpha, score] = self.getFundFactorExposure(PeriodFactorReturn, PeriodNAVReturn)
                ind = str(startTimeYear*100+startTimeMonth)+'-'+str(endTimeYear*100+endTimeMonth)
                AllPeriodFundFactorExposure.loc[ind,self.barra_ML_Return.columns.tolist()] = FundFactorExposure[0]
                AllPeriodFundFactorExposure.loc[ind,'alpha'] = FundAlpha
                AllPeriodScore.loc[ind,'score'] = score
            return AllPeriodFundFactorExposure, AllPeriodScore