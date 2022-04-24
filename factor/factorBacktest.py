# -*- coding: utf-8 -*-
"""
Created on Mon Jan  14 16:02:51 2020

@author: Wang
"""
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, GridSearchCV
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'

class SingleFactorBacktest(object):
    def __init__(self, factorName, factorExposure, price, tradeDateList, tradePoint='open',frequency = '日频'):
        # 因子名称，个股因子载荷，个股股价，交易时点，收益
        self.factorName = factorName
        # factorExposure默认一列为一只个股的不同时间载荷，一行为一个时点的不同个股载荷
        self.factorExposure = factorExposure
        # price格式为dataframe格式
        self.price = price
        self.tradePoint = tradePoint
        self.tradableDateList = tradeDateList
        self.frequency = frequency
        self.rts = self.price / self.price.shift(1) - 1
        self.rtsRank = self.rts.rank(method='dense', axis=1)
    def analyze(self, layerNum=10, positionPct=0.1, turnoverLimit=0.4):
        print('Start Backtest for %s' % self.factorName)
        self.factorRank = self.factorExposure.rank(axis=1, method='dense')
        # 重新生成factorRank，排除不可交易因子值
        self.generateFactorRank()
        # 分层组合回测
        self.groupBacktest(layerNum)
        # 计算因子IC
        self.ICBacktest()
        # 生成多空仓仓位
        self.generateLongShortPosition(positionPct, turnoverLimit)
        # 计算多空仓表现
        self.calcLongShortPerformance()
        # 计算纯多仓位回测情况
        self.pureLongBacktest()
        # 计算多空仓位回测情况
        # self.longShortBacktest()
    def generateFactorRank(self):
        # 本函数根据价格矩阵调整因子矩阵，只保留可交易的(时间，个股)因子数据
        # 默认self.price和self.factorRank矩阵大小相同
        nanPriceDF = self.price.fillna(0).astype(bool)
        self.factorExposure = pd.DataFrame(np.array(self.factorExposure) * np.array(nanPriceDF),
                                           index=self.factorExposure.index, columns=self.factorExposure.columns)
        # zeroFactorExposure用于判断每行中的0的位置，若不是0则为nan，若是0则为0，很奇怪
        # zeroFactorExposure = self.factorExposure[self.factorExposure == 0].astype(bool)
        # 对每行进行搜索，若全行均为0则不管，若部分为0则确认为不可交易，填充0为nan
        nanfactorExposure = self.factorExposure.copy()
        for ind in self.factorExposure.index:
            if self.factorExposure.loc[ind,:].sum() != 0:
                nanfactorExposure.loc[ind,nanfactorExposure.loc[ind,:]==0] = np.nan
        # 因子值排名，rank函数的参数中axis = 1即为以行为全样本集进行排名，axis = 0即为以列为全样本集进行排名
        self.factorRank = nanfactorExposure.rank(axis=1, method='dense')

    def groupBacktest(self,layerNum):
        print('Hierarchical Bakctest for %s' % self.factorName)
        # 计算分层个股的收益率
        self.calcGroupDailyRts(layerNum)
        # 画图展示分层个股的收益率
        self.plotGroupBacktestFigure()
    def getTradeDateList(self,frequency):
        TradeDateList = []
        if frequency == '日频':
            TradeDateList = self.tradableDateList
        elif frequency == '周频':
            # 若周频交易，选择每周一交易
            for time in self.tradableDateList:
                if time.day_of_week == 1:
                    TradeDateList.append(time)
        return TradeDateList
    def calcGroupDailyRts(self,layerNum):
        self.layerNum = layerNum
        groupCumRts = pd.DataFrame(index=self.factorRank.index)
        groupRts = pd.DataFrame(index=self.factorRank.index)
        print('for loop for calcGroupRts function')
        for layerIndex in tqdm(range(1,layerNum + 1)):
            groupName = 'group%s' % layerIndex
            groupPosition = pd.DataFrame(data=np.zeros(self.factorRank.shape),
                                         index = self.factorRank.index,columns = self.factorRank.columns)
            # 根据时点上排序个股数量决定分组情况
            # 当因子值不足以分为layerNum个组时，本方法失效
            if self.factorRank.max(skipna = True).max(skipna = True) < layerIndex:
                print('Number of Layers Excess Number of Ranks.')
                return
            elif self.factorRank.max(skipna = True).max(skipna = True) == layerIndex:
                groupPosition[self.factorRank == layerIndex] = 1
            else:
                # 日频换仓
                # sub函数用于查询和替换
                # 个股的factorRank数值越高，对应的layerIndex越大
                groupPosition[(self.factorRank.sub(self.factorRank.max(axis = 1)*(layerIndex-1)/layerNum,axis = 0) > 0)&
                              (self.factorRank.sub(self.factorRank.max(axis = 1)*layerIndex/layerNum,axis = 0) <= 0)] = 1
                # 周频换仓
                tradeDateList = self.getTradeDateList(self.frequency)
                tradeDateIndList = []
                tradeDateIndDict = {}
                for time in tradeDateList:
                    tradeDateIndList.append(self.tradableDateList.index(time))
                for timeInd in groupPosition.index:
                    tradeDateIndDict[timeInd] = [ind for ind in tradeDateIndList if ind - timeInd <= 0][-1]
                    groupPosition.loc[timeInd, :] = groupPosition.loc[tradeDateIndDict[timeInd], :]
            # 0316 如果出现了在某段时间内，所有个股都没有因子值，那么这段时间内的groupPosition应当清零
            for timeInd in groupPosition.index:
                if groupPosition.loc[timeInd,:].max(axis = 0) == groupPosition.loc[timeInd,:].min(axis = 0):
                    groupPosition.loc[timeInd, :] = 0
            # groupRts[groupName] = np.hstack((0, np.nanmean(groupPosition.iloc[:-1, :].values * self.rts.iloc[1:, :].values, axis=1)))
            groupRts[groupName] = np.hstack((0,0, np.nanmean(groupPosition.iloc[:-2, :].values * self.rts.iloc[2:, :].values, axis=1)))
            groupCumRts[groupName] = (1+groupRts[groupName]).cumprod()-1
        # 判断该因子值与收益率是正相关or负相关
        # group1的期末收益率小于group10时，factorMode=1时，载荷越大个股未来收益越高
        if groupCumRts.iloc[-1,0] < groupCumRts.iloc[-1,-1]:
            self.factorMode = 1
        else:
            self.factorMode = -1
        self.groupRts = groupRts
        self.groupCumRts = groupCumRts
    def plotGroupBacktestFigure(self):
        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('%s Hierarchical Backtest' % self.factorName)
        ax1.bar(self.groupCumRts.columns, 100 * self.groupCumRts.iloc[-1], color='blue', edgecolor='black')
        ax1.set_ylabel('cum returns (%)')
        ax1.set_title('%s Hierarchical Cum Returns Bar' % self.factorName)
        for groupIdx in range(self.groupRts.shape[1]):
            ax2.plot(self.groupCumRts.index, 100 * self.groupCumRts['group%s' % (groupIdx + 1)], linewidth=3)
        ax2.legend(self.groupCumRts.columns)
        ax2.set_title('%s Hierarchical Cum Returns Plot' % self.factorName)
        ax2.set_ylabel('cum returns (%)')
        ticks = np.arange(0, self.groupCumRts.shape[0], 90)
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(labels=self.groupCumRts.index[ticks], rotation=45)
        plt.show()
    def ICBacktest(self):
        print('IC Bakctest for %s' % self.factorName)
        self.calcIC()
        self.calcRankIC()
        print('%s IC performance is below:' % self.factorName)
        ICPerformance = pd.DataFrame(index=[0], columns=['IC_mean', 'ICIR', 'IC_tValue', 'rankIC_mean'])
        ICPerformance['IC_mean'] = round(self.IC.mean(),4)
        ICPerformance['ICIR'] = round(self.IC.mean() / self.IC.std(),4)
        # T统计量意义何在，我也不太清楚
        t_stats = stats.ttest_1samp(self.IC, 0)
        ICPerformance['IC_tValue'] = round(t_stats.statistic, 4)
        ICPerformance['rankIC_mean'] = round(self.rankIC.mean(),4)
        # 将'IC_mean'列设为index
        ICPerformance = ICPerformance.set_index(['IC_mean'])
        ICPerformance.index.name = None
        ICPerformance.columns.name = 'IC_mean'
        print(ICPerformance)
        self.ICPerformance = ICPerformance
        self.plotICFigures()
    def calcIC(self):
        IC = pd.Series(index = self.rts.index[1:], dtype='float64', name=self.factorName)
        for dateInd in range(len(self.rts.index[2:])):
            # 对每个时刻，计算当期因子值和第二天的个股收益的相关系数，得到二维矩阵
            # 如果价格取开盘价，则以第二天开盘价买入，第三天开盘价卖出
            # 如果价格取收盘价，则以当天收盘价买入，第二天收盘价卖出
            if self.tradePoint == 'close':
                corrDT = pd.DataFrame(list(zip(self.factorExposure.iloc[dateInd],self.rts.iloc[dateInd+1]))).dropna()
            elif self.tradePoint == 'open':
                corrDT = pd.DataFrame(list(zip(self.factorExposure.iloc[dateInd],self.rts.iloc[dateInd+2]))).dropna()
            # 0211 15:30得出结论，不能用当日因子与次日收益率做相关系数，因为次日收益率是当日开盘价买入，次日开盘价抛出，计算因子载荷时使用了未来信息
            # corrDT = pd.DataFrame(list(zip(self.factorExposure.iloc[dateInd],self.rts.iloc[dateInd+1]))).dropna()
            corrMat = corrDT.corr()
            IC[dateInd] = 0 if corrMat.isna().iloc[0,1] else corrMat.iloc[0, 1]
        # 0419 日频因子预测周频收益，周频调仓的话，计算IC的方式改为当期因子值和未来一周的个股收益的相关系数
        tradeDateList = self.getTradeDateList(self.frequency)
        IC = pd.Series(index = tradeDateList[1:], dtype='float64', name=self.factorName)
        for dateInd in tqdm(range(len(tradeDateList)-2)):
            currDate,currDateInd = tradeDateList[dateInd],self.rts.index.tolist().index(tradeDateList[dateInd])
            nextTradeDate,nextTradeDateInd = tradeDateList[dateInd+1],self.rts.index.tolist().index(tradeDateList[dateInd+1])
            currFactor = self.factorExposure.iloc[currDateInd]
            periodRts = (1+np.array(self.rts.iloc[currDateInd+2:nextTradeDateInd+2])).cumprod(axis = 0)-1
            corrDT = pd.DataFrame(list(zip(currFactor,periodRts[-1,:]))).dropna()
            corrMat = corrDT.corr()
            IC[dateInd] = 0 if corrMat.isna().iloc[0, 1] else corrMat.iloc[0, 1]
        self.IC = IC
    def calcRankIC(self):
        # 初始化IC序列，提前规定好了数值类型为float64
        ICRank = pd.Series(index=self.rts.index[1:], dtype='float64')
        for dateInd in range(len(self.rts.index[2:])):
            if self.tradePoint == 'close':
                corrDT = pd.DataFrame(list(zip(self.factorRank.iloc[dateInd],self.rts.iloc[dateInd+1]))).dropna()
            elif self.tradePoint == 'open':
                corrDT = pd.DataFrame(list(zip(self.factorRank.iloc[dateInd],self.rts.iloc[dateInd+2]))).dropna()
            # 0211 15:30得出结论，不能用当日因子与次日收益率做相关系数，因为次日收益率是当日开盘价买入，次日开盘价抛出，计算因子载荷时使用了未来信息
            # corrDT = pd.DataFrame(list(zip(self.factorRank.iloc[dateInd], self.rts.iloc[dateInd + 1]))).dropna()
            corrMat = corrDT.corr()
            ICRank[dateInd] = 0 if corrMat.isna().iloc[0,1] else corrMat.iloc[0,1]
        # 0419 日频因子预测周频收益，周频调仓的话，计算IC的方式改为当期因子值和未来一周的个股收益的相关系数
        tradeDateList = self.getTradeDateList(self.frequency)
        ICRank = pd.Series(index = tradeDateList[1:], dtype='float64', name=self.factorName)
        for dateInd in range(len(tradeDateList[2:])):
            currDate,currDateInd = tradeDateList[dateInd],self.rts.index.tolist().index(tradeDateList[dateInd])
            nextTradeDate,nextTradeDateInd = tradeDateList[dateInd+1],self.rts.index.tolist().index(tradeDateList[dateInd+1])
            currFactor = self.factorRank.iloc[currDateInd]
            periodRts = (1+np.array(self.rts.iloc[currDateInd+2:nextTradeDateInd+2])).cumprod(axis = 0)-1
            corrDT = pd.DataFrame(list(zip(currFactor,periodRts[-1,:]))).dropna()
            corrMat = corrDT.corr()
            ICRank[dateInd] = 0 if corrMat.isna().iloc[0, 1] else corrMat.iloc[0, 1]
        self.rankIC = ICRank

    def plotICFigures(self):
        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(14, 10))
        # 绘制IC曲线，添加一条IC均值的水平线
        ax1.plot(self.IC.index, self.IC, linewidth=3, label='IC curve')
        ax1.axhline(y=self.IC.mean(), color='g', linewidth=3, linestyle='-', label='IC mean')
        ax1.legend()
        ax1.set_title('%s IC curve' % self.factorName)
        ticks = np.arange(0, self.IC.shape[0], 90)
        #ax1.set_xticks(ticks)
        #ax1.set_xticklabels(labels=self.IC.index[ticks], rotation=45)
        # 绘制rankIC曲线，添加一条rankIC均值的水平线
        ax2.plot(self.rankIC.index, self.rankIC, linewidth=3, label='rankIC curve')
        ax2.axhline(y=self.rankIC.mean(), color='g', linewidth=3, linestyle='-', label='rankIC mean')
        ax2.legend()
        ax2.set_title('%s rankIC curve' % self.factorName)
        ticks = np.arange(0, self.rankIC.shape[0], 90)
        #ax2.set_xticks(ticks)
        #ax2.set_xticklabels(labels=self.rankIC.index[ticks], rotation=45)
        plt.subplots_adjust(left=0.125, right=0.9, top=0.92, wspace=0.5, hspace=0.5)
        plt.show()

    def generateLongShortPosition(self, positionPct, turnoverLimit):
        positionStockNum = (self.factorRank.notnull().sum(axis=1) * positionPct).astype(int)
        upperRank, lowerRank = self.generateUpperLowerRank(positionStockNum)

        upperPosition = self.factorRank.sub(upperRank, axis=0) >= 0
        lowerPosition = self.factorRank.sub(lowerRank, axis=0) <= 0
        # 因为因子载荷预测的是T+1日开盘至T+2日开盘的收益，所以代表的是T+1日的持仓
        upperPosition = upperPosition.shift(1).fillna(0).astype(bool)
        lowerPosition = lowerPosition.shift(1).fillna(0).astype(bool)
        # 因子载荷与未来收益率成正比时，因子载荷高的个股组成upperPosition，因子载荷低的个股组成lowerPosition
        if self.factorMode == 1:
            self.longPosition = upperPosition
            self.shortPosition = lowerPosition
        else:
            self.longPosition = lowerPosition
            self.shortPosition = upperPosition
        # self.reduceTurnover(positionStockNum, turnoverLimit)

    def reduceTurnover(self, positionStockNum, turnoverLimit):
        timestampList = self.factorRank.index.tolist()
        oldLongPosition = self.longPosition.iloc[0].copy()
        oldShortPosition = self.shortPosition.iloc[0].copy()
        for timestampIdx in tqdm(range(1, len(timestampList)), desc='Start Reducing Turnover:'):
            newLongPosition = self.longPosition.iloc[timestampIdx].copy()
            newShortPosition = self.shortPosition.iloc[timestampIdx].copy()
            turnoverThreshold = int(turnoverLimit * positionStockNum[timestampIdx-1])
            if (oldLongPosition ^ newLongPosition).sum() >= turnoverThreshold:
                # 看oldLongPosition在下一期factorExposure的因子排序，这里默认了因子值大的买入
                olgPositionNewFactorExposure = self.factorExposure.loc[timestampIdx]*oldLongPosition
                newPositionNewFactorExposure = self.factorExposure.loc[timestampIdx]*newLongPosition
                if self.factorMode == 1:
                    toReserveStockLoc = olgPositionNewFactorExposure[(olgPositionNewFactorExposure.sub(np.percentile(olgPositionNewFactorExposure,1-turnoverLimit))>=0)
                                                                     &(oldLongPosition !=0)].astype(bool)
                    toIgnoreStockLoc = newPositionNewFactorExposure[(newPositionNewFactorExposure.sub(np.percentile(newPositionNewFactorExposure,1-turnoverLimit))<=0)
                                                                     &(newLongPosition !=0)].astype(bool)
                else:
                    toReserveStockLoc = olgPositionNewFactorExposure[(olgPositionNewFactorExposure.sub(
                        np.percentile(olgPositionNewFactorExposure, 1 - turnoverLimit)) <= 0)
                                                                     & (oldLongPosition != 0)].astype(bool)
                    toIgnoreStockLoc = newPositionNewFactorExposure[(newPositionNewFactorExposure.sub(
                        np.percentile(newPositionNewFactorExposure, 1 - turnoverLimit)) >= 0)
                                                                    & (newLongPosition != 0)].astype(bool)

                newLongPosition.iloc[toReserveStockLoc.index] = 1
                newLongPosition.iloc[toIgnoreStockLoc.index] = 0
                self.longPosition.iloc[timestampIdx] = newLongPosition
            if (oldShortPosition ^ newShortPosition).sum() >= turnoverThreshold:
                # 看oldLongPosition在下一期factorExposure的因子排序，这里默认了因子值大的买入
                olgPositionNewFactorExposure = self.factorExposure.loc[timestampIdx] * oldShortPosition
                newPositionNewFactorExposure = self.factorExposure.loc[timestampIdx] * newShortPosition
                if self.factorMode == 1:
                    toReserveStockLoc = olgPositionNewFactorExposure[(olgPositionNewFactorExposure.sub(
                        np.percentile(olgPositionNewFactorExposure, turnoverLimit)) <= 0)
                                                                     & (oldShortPosition != 0)].astype(bool)
                    toIgnoreStockLoc = newPositionNewFactorExposure[(newPositionNewFactorExposure.sub(
                        np.percentile(newPositionNewFactorExposure, turnoverLimit)) >= 0)
                                                                    & (newShortPosition != 0)].astype(bool)
                else:
                    toReserveStockLoc = olgPositionNewFactorExposure[(olgPositionNewFactorExposure.sub(
                        np.percentile(olgPositionNewFactorExposure, turnoverLimit)) <= 0)
                                                                     & (oldShortPosition != 0)].astype(bool)
                    toIgnoreStockLoc = newPositionNewFactorExposure[(newPositionNewFactorExposure.sub(
                        np.percentile(newPositionNewFactorExposure, 1 - turnoverLimit)) >= 0)
                                                                    & (newShortPosition != 0)].astype(bool)

                newShortPosition.iloc[toReserveStockLoc.index] = 1
                newShortPosition.iloc[toIgnoreStockLoc.index] = 0
                self.shortPosition.iloc[timestampIdx] = newShortPosition
            # # 昊哥方法，基于要删除与要添加的元素做修改
            # if (oldLongPosition ^ newLongPosition).sum() >= turnoverThreshold:
            #     toDeleteStockLoc = np.where((oldLongPosition==1) & (newLongPosition==0))[0]
            #     toDeleteFactorExposure = self.factorExposure.iloc[timestampIdx, toDeleteStockLoc]
            #     toDeleteFactorRank = toDeleteFactorExposure.rank(method='min')
            #     toAddStockLoc = np.where((oldLongPosition==0) & (newLongPosition==1))[0]
            #     toAddFactorExposure = self.factorExposure.iloc[timestampIdx, toAddStockLoc]
            #     toAddFactorRank = toAddFactorExposure.rank(method='min')
            #     if self.factorMode == 1:
            #         toReserveStockLoc = toDeleteStockLoc[toDeleteFactorRank >= toAddFactorRank.shape[0] -
            #                                              turnoverThreshold / 2]
            #         toIgnoreStockLoc = toAddStockLoc[toAddFactorRank <= turnoverThreshold / 2]
            #     else:
            #         toReserveStockLoc = toDeleteStockLoc[toDeleteFactorRank <= turnoverThreshold / 2]
            #         toIgnoreStockLoc = toAddStockLoc[toAddFactorRank >= toDeleteFactorRank.shape[0] -
            #                                          turnoverThreshold / 2]
            #     newLongPosition.iloc[toReserveStockLoc] = 1
            #     newLongPosition.iloc[toIgnoreStockLoc] = 0
            #     self.longPosition.iloc[timestampIdx] = newLongPosition
            # if (oldShortPosition ^ newShortPosition).sum() >= turnoverThreshold:
            #     toDeleteStockLoc = np.where((oldShortPosition == 0) & (newShortPosition == 1))[0]
            #     toDeleteFactorExposure = self.factorExposure.iloc[timestampIdx, toDeleteStockLoc]
            #     toDeleteFactorRank = toDeleteFactorExposure.rank(method='min')
            #     toAddStockLoc = np.where((oldShortPosition == 0) & (newShortPosition == 1))[0]
            #     toAddFactorExposure = self.factorExposure.iloc[timestampIdx, toAddStockLoc]
            #     toAddFactorRank = toAddFactorExposure.rank(method='min')
            #     if self.factorMode == 1:
            #         toReserveStockLoc = toDeleteStockLoc[toDeleteFactorRank <= turnoverThreshold / 2]
            #         toIgnoreStockLoc = toAddStockLoc[toAddFactorRank >= toDeleteFactorRank -
            #                                          turnoverThreshold / 2]
            #     else:
            #         toReserveStockLoc = toDeleteStockLoc[toDeleteFactorRank >= toAddFactorRank.shape[0] -
            #                                              turnoverThreshold / 2]
            #         toIgnoreStockLoc = toAddStockLoc[toAddFactorRank <= turnoverThreshold / 2]
            #     newShortPosition.iloc[toReserveStockLoc] = 1
            #     newShortPosition.iloc[toIgnoreStockLoc] = 0
            #     self.shortPosition.iloc[timestampIdx] = newShortPosition
            oldLongPosition = newLongPosition.copy()
            oldShortPosition = newShortPosition.copy()

    def generateUpperLowerRank(self, positionStockNum):
        upperRank = np.zeros(self.factorRank.shape[0])
        lowerRank = np.zeros(self.factorRank.shape[0])

        print('for loop for generateUpperLowerRank function')
        for dateIdx in tqdm(range(self.factorRank.shape[0])):
            # numOfRanks是在dateIdx时点，因子值非空的个股数目，如果为0则后续无法运行
            numOfRanks = self.factorRank.iloc[dateIdx].value_counts()
            numOfRanks = numOfRanks.reindex(index=numOfRanks.index.sort_values(ascending=False))
            if len(numOfRanks) == 0:
                upperRank[dateIdx] = np.NaN
                lowerRank[dateIdx] = np.NaN
                print('No rank for date %s'% self.tradableDateList[dateIdx])
                continue
            upperStopRankIdx = 0
            upperNum = numOfRanks.iloc[upperStopRankIdx]
            while (upperNum < positionStockNum[dateIdx]):
                upperStopRankIdx += 1
                upperNum += numOfRanks.iloc[upperStopRankIdx]
            upperRank[dateIdx] = numOfRanks.index[upperStopRankIdx]

            lowerStopRankIdx = -1
            lowerNum = numOfRanks.iloc[lowerStopRankIdx]
            while (lowerNum < positionStockNum[dateIdx]):
                lowerStopRankIdx += -1
                lowerNum += numOfRanks.iloc[lowerStopRankIdx]
            lowerRank[dateIdx] = numOfRanks.index[lowerStopRankIdx]
            # if lowerRank[dateIdx] < 10:
            #     print('with less than 10 stocks for date %s' % self.tradableDateList[dateIdx])
        # upperRank, lowerRank代表着每个工作日做多和做空个股的个数
        return upperRank, lowerRank

    def calcLongShortPerformance(self):
        # 0328 计算考虑交易费率的累计收益
        stampTaxRate = 0
        upperRts = pd.Series(data=np.zeros(self.longPosition.shape[0]), index=self.tradableDateList)
        lowerRts = pd.Series(data=np.zeros(self.longPosition.shape[0]), index=self.tradableDateList)
        longTurnover = pd.Series(data=np.zeros(self.longPosition.shape[0]), index=self.tradableDateList)
        shortTurnover = pd.Series(data=np.zeros(self.longPosition.shape[0]), index=self.tradableDateList)

        # oldLongPosition = self.longPosition.iloc[0]
        # oldShortPosition = self.shortPosition.iloc[0]
        # wrongLongDateSituation = pd.DataFrame()
        # wrongShortDateSituation = pd.DataFrame()
        # # 0403 发现这个方案和之前的方案没有区别，都是等仓位买入，而非真正意义上的等权重，因此不适用
        # for dateIdx in tqdm(range(1,self.longPosition.shape[0])):
        #     newLongPosition = self.longPosition.iloc[dateIdx]
        #     newShortPosition = self.shortPosition.iloc[dateIdx]
        #     if sum(newLongPosition) >10 and sum(newShortPosition) >10:
        #         longTurnoverPosition = oldLongPosition ^ newLongPosition
        #         shortTurnoverPosition = oldShortPosition ^ newShortPosition
        #         longTurnover.iloc[dateIdx - 1] = longTurnoverPosition.sum() / oldLongPosition.sum()
        #         shortTurnover.iloc[dateIdx - 1] = shortTurnoverPosition.sum() / oldShortPosition.sum()
        #         # # 0329 由于2013-2015年部分时间的收益率出现了inf的情况，在这里我们对复权价格数据进行检查
        #         # currPrice,lastPrice = self.price.iloc[dateIdx],self.price.iloc[dateIdx-1]
        #         # currReturnRate = np.array(currPrice) / np.array(lastPrice) - 1
        #         # oldLongPosition = np.array(oldLongPosition) * (np.abs(currReturnRate) <= 0.1)
        #         # oldShortPosition = np.array(oldShortPosition) * (np.abs(currReturnRate) <= 0.1)
        #         lastDayLongCostArray = np.array(self.price.iloc[dateIdx-1])*(np.array(oldLongPosition))
        #         currDayLongGainArray = np.array(self.price.iloc[dateIdx])*(np.array(oldLongPosition - longTurnoverPosition * stampTaxRate))
        #         upperRts.iloc[dateIdx] = np.nanmean(currDayLongGainArray.astype(np.float64)-lastDayLongCostArray.astype(np.float64))/np.nanmean(lastDayLongCostArray.astype(np.float64))
        #
        #         lastDayShortCostArray = np.array(self.price.iloc[dateIdx - 1]) * (np.array(oldShortPosition))
        #         currDayShortGainArray = np.array(self.price.iloc[dateIdx]) * (np.array(oldShortPosition - shortTurnoverPosition * stampTaxRate))
        #         lowerRts.iloc[dateIdx] = np.nanmean(currDayShortGainArray.astype(np.float64)-lastDayShortCostArray.astype(np.float64))/np.nanmean(lastDayShortCostArray.astype(np.float64))
        #
        #         oldLongPosition = newLongPosition
        #         oldShortPosition = newShortPosition
        #     else:
        #         upperRts.iloc[dateIdx] = 0
        #         lowerRts.iloc[dateIdx] = 0
        #
        #     # 0329 输出收益率曲线的异常收益
        #     if np.abs(upperRts.iloc[dateIdx]) > 0.1:
        #         print('long Portfolio gets wrong in date %s'%self.tradableDateList[dateIdx])
        #         print(np.nanmean((currDayLongGainArray-lastDayLongCostArray).astype(np.float64)))
        #         print(np.nanmean(lastDayLongCostArray).astype(np.float64))
        #         print(upperRts.iloc[dateIdx])
        #         upperRts.iloc[dateIdx] = 0
        #         wrongLongDateSituation.loc[str(self.tradableDateList[dateIdx])+'oldLongPosition', self.longPosition.columns] = oldLongPosition.astype(int)
        #         wrongLongDateSituation.loc[str(self.tradableDateList[dateIdx])+'longTurnoverPosition', self.longPosition.columns] = longTurnoverPosition.astype(int)
        #         wrongLongDateSituation.loc[str(self.tradableDateList[dateIdx])+'lastPrice', self.longPosition.columns] = np.array(self.price.iloc[dateIdx-1])
        #         wrongLongDateSituation.loc[str(self.tradableDateList[dateIdx])+'currPrice', self.longPosition.columns] = np.array(self.price.iloc[dateIdx])
        #
        #     if np.abs(lowerRts.iloc[dateIdx]) > 0.1:
        #         print('short Portfolio gets wrong in date %s' % self.tradableDateList[dateIdx])
        #         print(np.nanmean((currDayShortGainArray-lastDayShortCostArray).astype(np.float64)))
        #         print(np.nanmean(lastDayShortCostArray).astype(np.float64))
        #         print(lowerRts.iloc[dateIdx])
        #         lowerRts.iloc[dateIdx] = 0
        #         wrongShortDateSituation.loc[str(self.tradableDateList[dateIdx])+'oldShortPosition', self.shortPosition.columns] = oldShortPosition.astype(int)
        #         wrongShortDateSituation.loc[str(self.tradableDateList[dateIdx])+'shortTurnoverPosition', self.shortPosition.columns] = shortTurnoverPosition.astype(int)
        #         wrongShortDateSituation.loc[str(self.tradableDateList[dateIdx])+'lastPrice', self.shortPosition.columns] = np.array(self.price.iloc[dateIdx - 1])
        #         wrongShortDateSituation.loc[str(self.tradableDateList[dateIdx])+'currPrice', self.shortPosition.columns] = np.array(self.price.iloc[dateIdx])
        # wrongLongDateSituation.to_csv(rootPath+'多头收益计算错误情况.csv',encoding = 'utf_8_sig')
        # wrongShortDateSituation.to_csv(rootPath+'空头收益计算错误情况.csv',encoding = 'utf_8_sig')

        # turnover定义：当日股票成交量占全部持股比例
        longTurnover = (self.longPosition ^ self.longPosition.shift(1)).sum(axis=1) / self.longPosition.sum(axis=1)
        shortTurnover = (self.shortPosition ^ self.shortPosition.shift(1)).sum(axis=1) / self.shortPosition.sum(axis=1)
        longTurnover.index = self.tradableDateList
        shortTurnover.index = self.tradableDateList
        # 0330 存在当天没有持仓的情况，turnover因此会有inf的项，填充为0
        for ind in longTurnover.index:
            if np.isinf(longTurnover.loc[ind]):
                longTurnover.loc[ind] = 1
            if np.isinf(shortTurnover.loc[ind]):
                shortTurnover.loc[ind] = 1
        # 0211 鉴于引入了self.tradableDateList，这里的index都改为self.tradableDateList
        upperRts = pd.Series(data=np.zeros(self.longPosition.shape[0]), index=self.tradableDateList)
        lowerRts = pd.Series(data=np.zeros(self.longPosition.shape[0]), index=self.tradableDateList)
        tradeDateList = self.getTradeDateList(self.frequency)
        for dateIdx in range(len(upperRts.index)):
            upperGroupName = 'group' + str(self.layerNum - 1)
            lowerGroupName = 'group1'
            if upperRts.index.tolist()[dateIdx] in tradeDateList:
                upperRts.iloc[dateIdx] = self.groupRts[upperGroupName][dateIdx] - stampTaxRate * longTurnover[dateIdx]
                lowerRts.iloc[dateIdx] = self.groupRts[lowerGroupName][dateIdx] - stampTaxRate * longTurnover[dateIdx]
            else:
                upperRts.iloc[dateIdx] = self.groupRts[upperGroupName][dateIdx]
                lowerRts.iloc[dateIdx] = self.groupRts[lowerGroupName][dateIdx]

        if self.factorMode == 1:
            longRts = upperRts
            shortRts = lowerRts
        else:
            shortRts = upperRts
            longRts = lowerRts
        self.longRts = longRts
        self.shortRts = shortRts
        self.longShortRts = longRts - shortRts
        self.longTurnover = longTurnover
        self.shortTurnover = shortTurnover
    def longShortBacktest(self):
        self.AnnualBacktest(self.longShortRts)

        print('Long Short Portfolio Backtest for %s' % self.factorName)
        performance = pd.DataFrame(index=[0],
                                   columns=['IC_mean', 'ICIR', 'rankIC_mean', 'cumRts(%)', 'annualVol(%)', 'maxDrawdown(%)', 'winRate(%)', 'SharpeRatio'])
        longShortNetValue = (1 + self.longShortRts).cumprod()
        performance['IC_mean'] = round(self.IC.mean(), 4)
        performance['ICIR'] = round(self.IC.mean() / self.IC.std(), 4)
        performance['rankIC_mean'] = round(self.rankIC.mean(), 4)

        performance['cumRts(%)'] = round(100 * (longShortNetValue.iloc[-1] - 1), 2)
        performance['annualRts(%)'] = round(100 * (longShortNetValue.iloc[-1]**(252/len(longShortNetValue)) - 1), 2)
        performance['annualVol(%)'] = round(100 * self.longShortRts.std() * np.sqrt(252), 2)
        expandingMaxNetValue = longShortNetValue.expanding().max()
        self.drawdown = longShortNetValue / expandingMaxNetValue - 1
        performance['maxDrawdown(%)'] = round(-100 * self.drawdown.min(), 2)
        performance['winRate(%)'] = round(100 * (self.longShortRts > 0).sum() / self.longShortRts.shape[0], 2)
        performance['SharpeRatio'] = round(performance['annualRts(%)'] / performance['annualVol(%)'], 2)
        performance['turnover'] = round(self.longTurnover.mean(), 2)
        self.performance = performance
        self.plotRtsFigure(self.longShortRts,'longShort cum returns')
        print(performance)

    def pureLongBacktest(self):
        self.AnnualBacktest(self.longRts)
        print('Pure Long Portfolio Backtest for %s' % self.factorName)
        performance = pd.DataFrame(index=[0],
                                  columns=['IC_mean', 'ICIR', 'rankIC_mean', 'cumRts(%)', 'annualVol(%)', 'maxDrawdown(%)', 'winRate(%)', 'SharpeRatio','turnover'])
        performance['IC_mean'] = round(self.IC.mean(), 4)
        performance['ICIR'] = round(self.IC.mean() / self.IC.std(), 4)
        performance['rankIC_mean'] = round(self.rankIC.mean(), 4)
        pureLongNetValue = (1 + self.longRts).cumprod()
        performance['cumRts(%)'] = round(100 * (pureLongNetValue.iloc[-1] - 1), 2)
        performance['annualRts(%)'] = round(100 * (pureLongNetValue.iloc[-1] ** (252 / len(pureLongNetValue)) - 1), 2)
        performance['annualVol(%)'] = round(100 * self.longRts.std() * (250**0.5), 2)
        expandingMaxNetValue = pureLongNetValue.expanding().max()
        self.drawdown = pureLongNetValue / expandingMaxNetValue - 1
        performance['maxDrawdown(%)'] = round(-100 * self.drawdown.min(), 2)
        performance['winRate(%)'] = round(100 * (self.longRts > 0).sum() / self.longRts.shape[0], 2)
        performance['turnover'] = round(self.longTurnover.mean(), 2)
        performance['SharpeRatio'] = round(performance['annualRts(%)'] / performance['annualVol(%)'], 2)

        self.performance = performance
        self.plotRtsFigure(self.longRts,'long cum returns')
        print(performance)
    def AnnualBacktest(self,rts):
        print('Annual Backtest for %s' % self.factorName)
        allYearList = list(set(rts.index.year))
        allYearList.sort()
        annualPerformance = pd.DataFrame(index = allYearList,columns=['cumRts(%)', 'annualVol(%)', 'maxDrawdown(%)', 'winRate(%)', 'SharpeRatio'])
        performance = pd.DataFrame(index=[0],
                                  columns=['cumRts(%)', 'annualVol(%)', 'maxDrawdown(%)', 'winRate(%)', 'SharpeRatio','turnover'])
        for year in annualPerformance.index:
            currYearDates = rts.index[rts.index.year == year]
            cumYearDates = rts.index[rts.index.year <= year]
            pureLongNetValue = (1 + rts[currYearDates]).cumprod()
            cumLongNetValue = (1 + rts[cumYearDates]).cumprod()
            annualPerformance.loc[year,'cumRts(%)'] = round(100 * (cumLongNetValue.iloc[-1] - 1), 2)
            annualPerformance.loc[year,'annualRts(%)'] = round(100 * (pureLongNetValue.iloc[-1] ** (252 / len(pureLongNetValue)) - 1), 2)
            annualPerformance.loc[year,'annualVol(%)'] = round(100 * rts[currYearDates].std() * np.sqrt(252), 2)
            if annualPerformance.loc[year,'annualVol(%)'] == 0:
                print('本年因子均在训练，全年因子载荷均为0')
            else:
                expandingMaxNetValue = pureLongNetValue.expanding().max()
                annualDrawdown = pureLongNetValue / expandingMaxNetValue - 1
                annualPerformance.loc[year,'maxDrawdown(%)'] = round(-100 * annualDrawdown.min(), 2)
                annualPerformance.loc[year,'winRate(%)'] = round(100 * (rts[currYearDates] > 0).sum() / rts[currYearDates].shape[0], 2)
                annualPerformance.loc[year, 'turnover'] = round(self.longTurnover[currYearDates].mean(),2)
                annualPerformance.loc[year,'SharpeRatio'] = round(annualPerformance.loc[year,'annualRts(%)'] / annualPerformance.loc[year,'annualVol(%)'], 2)
        print(annualPerformance)
        self.annualPerformance = annualPerformance
        # ##年化波动率
        # def annual_vol(nav, frequency=250):
        #     nav = nav.dropna()
        #     return nav.pct_change().std() * np.sqrt(frequency)
        #
        # ##年化收益率
        # def annual_return(nav, frequency=250):
        #     nav = nav.dropna()
        #     ##nav:净值序列
        #     return (nav[-1] / nav[0]) ** (frequency / len(nav)) - 1
        #
        # ##夏普比率
        # def sharp_ratio(nav, frequency=250, rf=0.0175):
        #     return (annual_return(nav, frequency) - rf) / annual_vol(nav, frequency)


    def plotRtsFigure(self,rts,rtsLabel):
        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 10))
        ax1.bar(self.longRts.index, 100 * self.longRts, color='red', label='single term long rts')
        ax1.set_ylabel('single term returns(%)')
        ax1.legend()
        ax1Right = ax1.twinx()
        ax1Right.plot(((1 + self.longShortRts).cumprod() - 1)*100, linewidth=3, label='long-short cum rts')
        ax1Right.plot(((1 + self.longRts).cumprod() - 1)*100, linewidth=3, label='long cum rts')
        ax1Right.plot(((1 + self.shortRts).cumprod() - 1)*100, linewidth=3, label='short cum rts')
        ax1Right.set_ylabel('cum returns(%)')
        ax1Right.legend()
        ticks = np.arange(0, self.longShortRts.shape[0], 90)
        #ax1.set_xticks(ticks)
        #ax1.set_xticklabels(labels=self.longShortRts.index[ticks])
        fig.autofmt_xdate(rotation=45)
        ax2.plot(((1 + rts).cumprod() - 1)*100, linewidth=3, color='red', label=rtsLabel)
        ax2.set_ylabel('cum returns(%)')
        ax2.legend()
        # ax2Right = ax2.twinx()
        # ax2Right.fill_between(self.drawdown.index, 100 * self.drawdown, 0, color='grey', label='drawdown')
        # ax2Right.set_ylabel('drawdown(%)')
        #ax2.set_xticks(ticks)
        #ax2.set_xticklabels(labels=self.longShortRts.index[ticks])
        # fig.autofmt_xdate(rotation=45)
        # ax2Right.legend()

        plt.suptitle('%s Backtest Performance: ' % self.factorName)
        plt.show()

    def plotTurnover(self):
        fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(20, 25))
        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.2, top=0.9, wspace=0.2, hspace=0.5)

        ax1.fill_between(self.longTurnover.index[1:], self.longTurnover.iloc[1:] * 100, 0, facecolor='pink', alpha=0.4)
        ticks = np.arange(0, self.longTurnover.shape[0], 90)
        #ax1.set_xticks(ticks)
        #ax1.set_xticklabels(labels=self.longTurnover.index[ticks + 1], rotation=45)
        ax1.set_title('Turnover on the Long Side')

        ax2.fill_between(self.shortTurnover.index[1:], self.shortTurnover.iloc[1:] * 100, 0, facecolor='green',
                         alpha=0.4)
        #ax2.set_xticks(ticks)
        #ax2.set_xticklabels(labels=self.longTurnover.index[ticks + 1], rotation=45)
        ax2.set_title('Turnover on the Short Side')

        ax3.fill_between(self.longTurnover.index[1:], self.longTurnover.iloc[1:] * 100, 0, facecolor='pink', alpha=0.4)
        ax3.fill_between(self.shortTurnover.index[1:], self.shortTurnover.iloc[1:] * 100, 0, facecolor='green',
                         alpha=0.4)
        ax3.legend(['Long', 'Short'])
        #ax3.set_xticks(ticks)
        #ax3.set_xticklabels(labels=self.longTurnover.index[ticks + 1], rotation=45)
        ax3.set_title('Turnover on Both Sides')
        plt.suptitle('Turnover Analysis')
        plt.show()
