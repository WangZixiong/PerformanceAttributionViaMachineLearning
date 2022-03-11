# -*- coding: utf-8 -*-
"""
Created on Mon Jan  14 16:02:51 2020

@author: Wang
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from tqdm import tqdm
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, GridSearchCV
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'

class SingleFactorBacktest(object):
    def __init__(self, factorName, factorExposure, price, tradeDateList, tradePoint='close'):
        # 因子名称，个股因子载荷，个股股价，交易时点，收益
        self.factorName = factorName
        # factorExposure默认一列为一只个股的不同时间载荷，一行为一个时点的不同个股载荷
        self.factorExposure = factorExposure
        # price格式为dataframe格式
        self.price = price
        self.tradePoint = tradePoint
        self.tradeDates = tradeDateList
        self.rts = self.price / self.price.shift(1) - 1
        self.rtsRank = self.rts.rank(method='dense', axis=1)
    def analyze(self, layerNum=10, positionPct=0.1, turnoverLimit=0.5):
        print('Start Backtest for %s' % self.factorName)
        # 因子值排名，rank函数的参数中axis = 1即为以行为全样本集进行排名，axis = 0即为以列为全样本集进行排名
        self.factorRank = self.factorExposure.rank(axis=1, method='dense')
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
    def groupBacktest(self,layerNum):
        print('Hierarchical Bakctest for %s' % self.factorName)
        # 计算分层个股的收益率
        self.calcGroupRts(layerNum)
        # 画图展示分层个股的收益率
        self.plotGroupBacktestFigure()
    def calcGroupRts(self,layerNum):
        # fac_pct = self.factorExposure.rank(axis=1, method='min', na_option='keep', ascending=True, pct=True)
        # stock_layer = fac_pct.apply(lambda x: pd.cut(x,
        #                                              bins=np.linspace(0, 1, layerNum + 1),
        #                                              right=True,
        #                                              labels=np.arange(1, layerNum + 1),
        #                                              include_lowest=True,
        #                                              ordered=False).astype(float),
        #                             axis=1)
        # layer_rts = pd.concat(
        #     [self.rts[stock_layer.shift() == i].mean(axis=1, skipna=True)
        #      for i in range(1, layerNum + 1)],
        #     axis=1
        # )
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
                # sub函数用于查询和替换
                # 个股的factorRank数值越高，对应的layerIndex越大
                groupPosition[(self.factorRank.sub(self.factorRank.max(axis = 1)*(layerIndex-1)/layerNum,axis = 0) > 0)&
                              (self.factorRank.sub(self.factorRank.max(axis = 1)*layerIndex/layerNum,axis = 0) <= 0)] = 1
            # groupRts[groupName] = np.hstack((0, np.nanmean(groupPosition.iloc[:-1, :].values * self.rts.iloc[1:, :].values, axis=1)))
            groupRts[groupName] = np.hstack((0,0, np.nanmean(groupPosition.iloc[:-2, :].values * self.rts.iloc[2:, :].values, axis=1)))
            groupCumRts[groupName] = (1+groupRts[groupName]).cumprod()-1
            # if layerIndex == 10:
            #     self.groupLongPosition = groupPosition
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
        ICPerformance['IC_mean'] = self.IC.mean()
        ICPerformance['ICIR'] = self.IC.mean() / self.IC.std()
        # T统计量意义何在，我也不太清楚
        t_stats = stats.ttest_1samp(self.IC, 0)
        # ICPerformance['IC_tValue'] = round(t_stats.statistic, 4)
        ICPerformance['rankIC_mean'] = self.rankIC.mean()
        # 下面这一段没看太懂，后面根据效果研究一下？？？？？？？？？？？
        ICPerformance = ICPerformance.set_index(['IC_mean'])
        ICPerformance.index.name = None
        ICPerformance.columns.name = 'IC_mean'

        print(ICPerformance)
        self.ICPerformance = ICPerformance
        self.plotICFigures()
    def calcIC(self):
        # 初始化IC序列，提前规定好了数值类型为float64，不知道是何目的
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
            if (oldLongPosition ^ newShortPosition).sum() >= turnoverThreshold:
                toDeleteStockLoc = np.where((oldLongPosition==1) & (newLongPosition==0))[0]
                toDeleteFactorExposure = self.factorExposure.iloc[timestampIdx, toDeleteStockLoc]
                toDeleteFactorRank = toDeleteFactorExposure.rank(method='min')
                toAddStockLoc = np.where((oldLongPosition==0) & (newLongPosition==1))[0]
                toAddFactorExposure = self.factorExposure.iloc[timestampIdx, toAddStockLoc]
                toAddFactorRank = toAddFactorExposure.rank(method='min')
                if self.factorMode == 1:
                    toReserveStockLoc = toDeleteStockLoc[toDeleteFactorRank >= toAddFactorRank.shape[0] -
                                                         turnoverThreshold / 2]
                    toIgnoreStockLoc = toAddStockLoc[toAddFactorRank <= turnoverThreshold / 2]
                else:
                    toReserveStockLoc = toDeleteStockLoc[toDeleteFactorRank <= turnoverThreshold / 2]
                    toIgnoreStockLoc = toAddStockLoc[toAddFactorRank >= toDeleteFactorRank.shape[0] -
                                                     turnoverThreshold / 2]
                newLongPosition.iloc[toReserveStockLoc] = 1
                newLongPosition.iloc[toIgnoreStockLoc] = 0
                self.longPosition.iloc[timestampIdx] = newLongPosition

            if (oldShortPosition ^ newShortPosition).sum() >= turnoverThreshold:
                toDeleteStockLoc = np.where((oldShortPosition == 0) & (newShortPosition == 1))[0]
                toDeleteFactorExposure = self.factorExposure.iloc[timestampIdx, toDeleteStockLoc]
                toDeleteFactorRank = toDeleteFactorExposure.rank(method='min')
                toAddStockLoc = np.where((oldShortPosition == 0) & (newShortPosition == 1))[0]
                toAddFactorExposure = self.factorExposure.iloc[timestampIdx, toAddStockLoc]
                toAddFactorRank = toAddFactorExposure.rank(method='min')
                if self.factorMode == 1:
                    toReserveStockLoc = toDeleteStockLoc[toDeleteFactorRank <= turnoverThreshold / 2]
                    toIgnoreStockLoc = toAddStockLoc[toAddFactorRank >= toDeleteFactorRank -
                                                     turnoverThreshold / 2]
                else:
                    toReserveStockLoc = toDeleteStockLoc[toDeleteFactorRank >= toAddFactorRank.shape[0] -
                                                         turnoverThreshold / 2]
                    toIgnoreStockLoc = toAddStockLoc[toAddFactorRank <= turnoverThreshold / 2]
                newShortPosition.iloc[toReserveStockLoc] = 1
                newShortPosition.iloc[toIgnoreStockLoc] = 0
                self.shortPosition.iloc[timestampIdx] = newShortPosition
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
        # upperRank, lowerRank代表着每个工作日做多和做空个股的个数
        return upperRank, lowerRank

    def calcLongShortPerformance(self):
        # # 0211 鉴于引入了self.tradeDates变量，这里的index都改为self.tradeDates
        upperRts = pd.Series(data=np.zeros(self.longPosition.shape[0]), index=self.tradeDates)
        lowerRts = pd.Series(data=np.zeros(self.longPosition.shape[0]), index=self.tradeDates)
        for dateIdx in range(len(upperRts.index)):
            # 0304 为了防止极端值不佳，我们挑选第2组和第9组进行比较
            upperGroupName = 'group'+str(self.layerNum-1)
            lowerGroupName = 'group2'
            upperRts.iloc[dateIdx] = self.groupRts[upperGroupName][dateIdx]
            lowerRts.iloc[dateIdx] = self.groupRts[lowerGroupName][dateIdx]
        if self.factorMode == 1:
            longRts = upperRts
            shortRts = lowerRts
        else:
            shortRts = upperRts
            longRts = lowerRts
        longShortRts = longRts - shortRts
        # turnover定义：当日股票成交量占全部持股比例
        longTurnover = (self.longPosition ^ self.longPosition.shift(1)).sum(axis = 1)/self.longPosition.sum(axis = 1)
        shortTurnover = (self.shortPosition ^ self.shortPosition.shift(1)).sum(axis=1) / self.shortPosition.sum(axis=1)
        longTurnover.index = self.tradeDates
        shortTurnover.index = self.tradeDates

        self.longRts = longRts
        self.shortRts = shortRts
        self.longShortRts = longShortRts
        self.longTurnover = longTurnover
        self.shortTurnover = shortTurnover
    def longShortBacktest(self):
        print('Long Short Portfolio Backtest for %s' % self.factorName)
        performance = pd.DataFrame(index=[0],
                                   columns=['cumRts(%)', 'annualVol(%)', 'maxDrawdown(%)', 'winRate(%)', 'SharpeRatio'])
        longShortNetValue = (1 + self.longShortRts).cumprod()
        performance['cumRts(%)'] = round(100 * (longShortNetValue.iloc[-1] - 1), 2)
        performance['annualRts(%)'] = round(100 * (longShortNetValue.iloc[-1]**(252/len(longShortNetValue)) - 1), 2)
        performance['annualVol(%)'] = round(100 * self.longShortRts.std() * ((237*250)**0.5), 2)
        expandingMaxNetValue = longShortNetValue.expanding().max()
        self.drawdown = longShortNetValue / expandingMaxNetValue - 1
        performance['maxDrawdown(%)'] = round(-100 * self.drawdown.min(), 2)
        performance['winRate(%)'] = round(100 * (self.longShortRts > 0).sum() / self.longShortRts.shape[0], 2)
        performance['SharpeRatio'] = round(self.longShortRts.mean(skipna= True) / self.longShortRts.std(), 4)

        self.cumRts,self.maxDrawdown = float(performance['cumRts(%)']),float(performance['maxDrawdown(%)'])
        self.winRate,self.SharpeRatio = float(performance['winRate(%)']),float(performance['SharpeRatio'])
        self.annualVol,self.annualRts = float(performance['annualVol(%)']),float(performance['annualRts(%)'])

        performance.set_index(['cumRts(%)'], inplace=True)
        performance.index.name = None
        performance.columns.name = 'cumRts(%)'
        self.performance = performance

        self.plotRtsFigure()
        self.plotTurnover()
        print(performance)

    def pureLongBacktest(self):
        print('Pure Long Portfolio Backtest for %s' % self.factorName)
        allYearList = list(set(self.longRts.index.year))
        allYearList.sort()
        annualPerformance = pd.DataFrame(index = allYearList,columns=['cumRts(%)', 'annualVol(%)', 'maxDrawdown(%)', 'winRate(%)', 'SharpeRatio'])
        performance = pd.DataFrame(index=[0],
                                  columns=['cumRts(%)', 'annualVol(%)', 'maxDrawdown(%)', 'winRate(%)', 'SharpeRatio','turnover'])
        for year in annualPerformance.index:
            currYearDates = self.longRts.index[self.longRts.index.year == year]
            pureLongNetValue = (1 + self.longRts[currYearDates]).cumprod()
            annualPerformance.loc[year,'cumRts(%)'] = round(100 * (pureLongNetValue.iloc[-1] - 1), 2)
            annualPerformance.loc[year,'annualRts(%)'] = round(100 * (pureLongNetValue.iloc[-1] ** (252 / len(pureLongNetValue)) - 1), 2)
            annualPerformance.loc[year,'annualVol(%)'] = round(100 * self.longRts[currYearDates].std() * (250**0.5), 2)
            if annualPerformance.loc[year,'annualVol(%)'] == 0:
                print('本年因子均在训练，全年因子载荷均为0')
            else:
                expandingMaxNetValue = pureLongNetValue.expanding().max()
                self.drawdown = pureLongNetValue / expandingMaxNetValue - 1
                annualPerformance.loc[year,'maxDrawdown(%)'] = round(-100 * self.drawdown.min(), 2)
                annualPerformance.loc[year,'winRate(%)'] = round(100 * (self.longRts[currYearDates] > 0).sum() / self.longRts[currYearDates].shape[0], 2)
                annualPerformance.loc[year, 'turnover'] = round(self.longTurnover[currYearDates].mean(),2)
                annualPerformance.loc[year,'SharpeRatio'] = round(self.longRts[currYearDates].mean() / self.longRts[currYearDates].std(), 4)
        # annualPerformance.to_csv(rootPath+'factor\\'+self.factorName+'annualPerformance.csv',encoding = 'utf_8_sig')
        print(annualPerformance)
        pureLongNetValue = (1 + self.longRts).cumprod()
        performance['cumRts(%)'] = round(100 * (pureLongNetValue.iloc[-1] - 1), 2)
        performance['annualRts(%)'] = round(
            100 * (pureLongNetValue.iloc[-1] ** (252 / len(pureLongNetValue)) - 1), 2)
        performance['annualVol(%)'] = round(100 * self.longRts.std() * (250**0.5), 2)
        expandingMaxNetValue = pureLongNetValue.expanding().max()
        self.drawdown = pureLongNetValue / expandingMaxNetValue - 1
        performance['maxDrawdown(%)'] = round(-100 * self.drawdown.min(), 2)
        performance['winRate(%)'] = round(100 * (self.longRts > 0).sum() / self.longRts.shape[0], 2)
        performance['turnover'] = round(self.longTurnover.mean(), 2)
        performance['SharpeRatio'] = round(performance['annualRts(%)'] / performance['annualVol(%)'], 4)

        self.performance = performance
        self.cumRts, self.maxDrawdown = float(performance['cumRts(%)']),  float(performance['maxDrawdown(%)'])
        self.annualVol,self.annualRts = float(performance['annualVol(%)']),float(performance['annualRts(%)'])
        self.winRate, self.SharpeRatio = float(performance['winRate(%)']), float(performance['SharpeRatio'])
        self.longTurnover = float(performance['turnover'])
        performance.set_index(['cumRts(%)'], inplace=True)
        performance.index.name = None
        performance.columns.name = 'cumRts(%)'
        self.plotRtsFigure()
        # self.plotTurnover()
        print(performance)

    def plotRtsFigure(self):
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
        ax2.plot(((1 + self.longRts).cumprod() - 1)*100, linewidth=3, color='red', label='long-short cum returns')
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
