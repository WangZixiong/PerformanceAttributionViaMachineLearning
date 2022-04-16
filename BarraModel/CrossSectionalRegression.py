# -*- coding: utf-8 -*-
"""
Created on Wen Mar 25 10:02:51 2022

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
import warnings
warnings.filterwarnings('ignore')
linearregressionparams = {
    'fit_intercept':True,
    'copy_X':True,
    'n_jobs':-1
}
ridgeparams = {
    'fit_intercept':True,
    'copy_X':True,
    'alpha':0.001
}
class CrossSectionalRegression():
    def __init__(self,OpenPriceData,barraExposure,industryRawDF,ML_factorDict):
        # 读取OpenPrice
        self.OpenPrice = OpenPriceData['openPrice']
        # 读取barraExposure，barraExposure时间跨度是20050104-20200316，个股对OpenPrice中的个股几乎全覆盖（差25只）
        self.barraExposure = barraExposure
        # 读取中信行业分类数据
        self.industryRawDF = industryRawDF
        # 读取机器学习因子
        self.ML_factorDict = ML_factorDict

        # 获取OpenPrice和barraExposure的共同时间与共同个股
        tradeTimeList = OpenPriceData['sharedInformation']['axis1Time']
        stockList = OpenPriceData['sharedInformation']['axis2Stock']
        rawBarraStockList = list(set(barraExposure.SecuCode))
        rawBarraTradeTimeList = list(set(barraExposure.MarketTime))
        industryStockList = industryRawDF.columns.tolist()
        industryTimeList = industryRawDF.index.tolist()
        BarraStockList = []
        for stock in rawBarraStockList:
            BarraStockList.append(self.switchBarraStockCode(stock))
        # barraStock和barraTime分别存储了全时间上的个股交集与时间交集
        self.barraStock = [i for i in stockList if i in BarraStockList and i in industryStockList]
        self.barraTime = [i for i in tradeTimeList if i in rawBarraTradeTimeList and i in industryTimeList]
        self.barraStock.sort()
        self.barraTime.sort()

    # 获取行业因子、风格因子与国家因子收益
    def getFactorReturn(self,barraFactorList):
        # 获取个股收益率,每日个股收益为T+2日开盘/T+1日开盘，作为T+1期收益
        self.OpenPriceReturn = self.OpenPrice.shift(-2) / self.OpenPrice.shift(-1)-1
        # 截面回归获取当日因子收益率
        # 两种方法都是个股收益率与因子载荷回归，第一种是得到因子收益率，第二种是得到个股权重矩阵
        FactorReturn,scoreDF,stockNumDF = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
        for date in tqdm(self.barraTime):
            # 根据barraFactorList获取因子载荷
            rawCurrRiskFactorMatrix = self.getRiskFactorMatrix(barraFactorList,date)
            rawCurrIndustryFactorMatrix = self.getIndustryFactorMatrix(date)
            # 合并风格因子与行业因子矩阵，可能存在个股有行业因子载荷但没有风格因子载荷
            rawCurrRiskFactorMatrix.loc[rawCurrIndustryFactorMatrix.index,rawCurrIndustryFactorMatrix.columns] = rawCurrIndustryFactorMatrix
            rawCurrOpenPriceReturn = self.OpenPriceReturn.loc[date,self.barraStock]

            # 删除当下时间缺省的个股的特征与收益
            riskFactorStockList = []
            openPriceStockList = []
            industryStockList = rawCurrIndustryFactorMatrix.index.tolist()
            for stock in rawCurrRiskFactorMatrix.index:
                if np.isnan(sum(rawCurrRiskFactorMatrix.loc[stock,:])) == 0 and stock in industryStockList:
                    riskFactorStockList.append(stock)
            for stock in rawCurrOpenPriceReturn.index:
                if np.isnan(rawCurrOpenPriceReturn[stock]) == 0:
                    openPriceStockList.append(stock)
            finalStockList = [i for i in riskFactorStockList if i in openPriceStockList]
            currRiskFactorMatrix = rawCurrRiskFactorMatrix.loc[finalStockList,:]
            currOpenPriceReturn = rawCurrOpenPriceReturn[finalStockList]

            # 计算最终使用个股个数,为了防止用的股票数太少
            # stockNumDF.loc[date,'availableStockNum'] = len(finalStockList)

            # 正交化、标准化处理因子载荷
            # 这一步先省略，标准化聚宽做好了，正交化等机器学习因子再说吧
            # self.preprocessingFactor(barraFactorList,date)

            # 当日因子收益率为K*1向量，K为因子数，1为一天收益
            [currFactorReturn,currCountryReturn,score] = self.getCrossSectionReturn(currRiskFactorMatrix,currOpenPriceReturn,ridgeparams)
            FactorReturn.loc[date,currRiskFactorMatrix.columns] = currFactorReturn
            FactorReturn.loc[date, 'country'] = currCountryReturn
            scoreDF.loc[date,'score'] = score
        return [FactorReturn,scoreDF]
    def getRiskFactorMatrix(self,barraFactorList,date):
        factorMatrixTimeGroup = self.barraExposure[self.barraExposure.MarketTime == date]
        self.factorDict = {}
        currFactorMatrix = pd.DataFrame(index=self.barraStock)
        for factor in barraFactorList:
            # currFactorRawMatrix = factorMatrixTimeGroup.loc[:,factor]
            for ind in factorMatrixTimeGroup.index:
                currStock = self.switchBarraStockCode(factorMatrixTimeGroup.loc[ind,'SecuCode'])
                if currStock in currFactorMatrix.index:
                    currFactorMatrix.loc[currStock,factor] = factorMatrixTimeGroup.loc[ind,factor]
        return currFactorMatrix
    def getIndustryFactorMatrix(self,date):
        currStockIndustry = self.industryRawDF.loc[date]
        industryDict = {'石油石化': 'CITICSPetroleumPetrochemical', '煤炭': 'CITICSCoal', '有色金属': 'CITICSNonFerrousMetal',
                        '钢铁': 'CITICSSteel', '基础化工': 'CITICSBasicChemical',
                        '电力及公用事业': 'CITICSElectricity', '建筑': 'CITICSArchitecture', '建材': 'CITICSBuildingMaterial',
                        '房地产': 'CITICSRealEstate', '交通运输': 'CITICSTransportation',
                        '银行': 'CITICSBank', '非银行金融': 'CITICSNonBankFinancial', '综合金融': 'CITICSNonBankFinancial',
                        '电子': 'CITICSElectronic', '通信': 'CITICSCommunication',
                        '计算机': 'CITICSComputer', '传媒': 'CITICSMedia', '轻工制造': 'CITICSLightManufacture',
                        '商贸零售': 'CITICSCommercialRetail', '消费者服务': 'CITICSConsumerService',
                        '纺织服装': 'CITICSTextileClothing', '食品饮料': 'CITICSFoodBeverage', '农林牧渔': 'CITICSAgriculture',
                        '医药': 'CITICSMedicine', '机械': 'CITICSMachine',
                        '电力设备及新能源': 'CITICSNewEnergy', '国防军工': 'CITICSMilitary', '汽车': 'CITICSAutomobile',
                        '家电': 'CITICSHomeAppliance', '综合': 'CITICSComposite'}
        # currStockIndustryBoolDF用于统计当下个股的行业分类
        currStockIndustryBoolDF = pd.DataFrame()
        # currIndustryList用于统计当下个股所在的所有行业，以寻找缺省行业
        # currIndustryList = []
        for stock in currStockIndustry.index:
            # 若stock有行业标签，说明该个股在此时已经上市了，若没有，则可能没上市，不一定是没获取到行业标签
            if currStockIndustry[stock] in industryDict:
                currIndustry = industryDict[currStockIndustry[stock]]
                currStockIndustryBoolDF.loc[stock,currIndustry] = 1
                # if currStockIndustry[stock] not in currIndustryList:
                #     currIndustryList.append(currStockIndustry[stock])
        currStockIndustryBoolDF = currStockIndustryBoolDF.fillna(0)
        return currStockIndustryBoolDF
    def getMLFactorMatrix(self,date,adjustedMLFactorDF):
        currMLFactorMatrix = adjustedMLFactorDF[adjustedMLFactorDF.index == date]
        return currMLFactorMatrix
    def preprocessingFactor(self,barraFactorList):
        # 为了防止出现因子载荷的共线性，使得解不唯一，故需要做正交化
        self.orthogonalize(barraFactorList)
        self.normalize(barraFactorList)

    def switchBarraStockCode(self,SecuCode):
        if SecuCode[0] == '6':
            return SecuCode+'.SH'
        else:
            return SecuCode + '.SZ'
    def getCrossSectionReturn(self,currRiskFactorMatrix,currOpenPriceReturn,params):
        model = sklearn.linear_model.Ridge(**params)
        reg = model.fit(currRiskFactorMatrix,currOpenPriceReturn)
        currFactorReturn,currCountryReturn,score = reg.coef_,reg.intercept_,reg.score(currRiskFactorMatrix,currOpenPriceReturn)
        allFactorReturn = list(currFactorReturn)
        return [allFactorReturn,currCountryReturn,score]
    # 本函数用于将MLFactor和barra因子正交化，通过多元回归观察该因子与barra各个因子的相关性
    # 输出两个矩阵：MLFactorRegressionResult表示每个时间截面上的回归结果
    # adjustedMLFactorDF表示正交化后的因子载荷残差项
    def orthogonalize(self,barraFactorList):
        allTime = [i for i in self.ML_factorDict['axis1Time'] if i in self.barraTime]
        MLFactorRegressionResult = pd.DataFrame(index = allTime)
        adjustedMLFactorDF = pd.DataFrame()
        for date in tqdm(allTime):
            # 根据barraFactorList获取因子载荷
            rawCurrRiskFactorMatrix = self.getRiskFactorMatrix(barraFactorList,date)
            rawCurrIndustryFactorMatrix = self.getIndustryFactorMatrix(date)
            rawCurrRiskFactorMatrix.loc[
                rawCurrIndustryFactorMatrix.index, rawCurrIndustryFactorMatrix.columns] = rawCurrIndustryFactorMatrix
            finalStockList = []
            for stock in rawCurrRiskFactorMatrix.index:
                if np.isnan(sum(rawCurrRiskFactorMatrix.loc[stock,:])) == 0 and stock in self.ML_factorDict['axis2Stock']:
                    finalStockList.append(stock)\
            # 整理机器学习因子矩阵
            # 寻找self.ML_factorDict中位于当下时点，指定个股的因子载荷矩阵
            if type(self.ML_factorDict['axis1Time']) == list:
                dateInd = self.ML_factorDict['axis1Time'].index(date)
            else:
                dateInd = self.ML_factorDict['axis1Time'].tolist().index(date)
            finalStockListInd = []
            MLFinalStockList = []
            for stock in finalStockList:
                if stock in self.ML_factorDict['axis2Stock']:
                    finalStockListInd.append(self.ML_factorDict['axis2Stock'].index(stock))
                    MLFinalStockList.append(stock)
            currMLFactor = self.ML_factorDict['stk_loading'].loc[dateInd,finalStockListInd]

            # 整理barra因子载荷矩阵
            # 0413为了防止出现该时点下有些行业无个股，导致本行业所在列均为0的情况，删除了求和为0的列
            CurrRiskFactorMatrix = pd.DataFrame()
            for col in rawCurrRiskFactorMatrix.columns:
                if sum(rawCurrRiskFactorMatrix.loc[:,col]) != 0:
                    CurrRiskFactorMatrix = rawCurrRiskFactorMatrix.loc[finalStockList,:]
            # model = sklearn.linear_model.LinearRegression(**linearregressionparams)
            model = sklearn.linear_model.Ridge(**ridgeparams)
            reg = model.fit(CurrRiskFactorMatrix, currMLFactor)
            currBeta, MLFactorAlpha, score = reg.coef_, reg.intercept_, reg.score(CurrRiskFactorMatrix,currMLFactor)
            # 检查是否还存在极端beta值
            currTimeStr = str(date.year*10000+date.month*100+date.day)
            if max(list(currBeta)) > 10:
                print(currTimeStr+'存在异常beta值')
            MLFactorRegressionResult.loc[date,CurrRiskFactorMatrix.columns] = list(currBeta)
            MLFactorRegressionResult.loc[date, 'alpha'] = MLFactorAlpha
            # 正交后，对机器学习因子取残差+截距项
            currAdjustedMLFactor = np.array(currMLFactor) - np.dot(np.array(CurrRiskFactorMatrix),currBeta)
            adjustedMLFactorDF.loc[date,MLFinalStockList] = currAdjustedMLFactor
        MLFactorRegressionResult.fillna(0,inplace=True)
        adjustedMLFactorDF.fillna(0,inplace=True)
        return MLFactorRegressionResult,adjustedMLFactorDF

    # 获取行业因子、风格因子与国家因子收益
    def getBarraPlusMLFactorReturn(self, adjustedMLFactorDF,barraFactorList):
        # 获取个股收益率,每日个股收益为T+2日开盘/T+1日开盘，作为T+1期收益
        self.OpenPriceReturn = self.OpenPrice.shift(-2) / self.OpenPrice.shift(-1) - 1
        # 截面回归获取当日因子收益率
        # 两种方法都是个股收益率与因子载荷回归，第一种是得到因子收益率，第二种是得到个股权重矩阵
        FactorReturn, scoreDF, stockNumDF = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for date in tqdm(self.barraTime):
            # 根据barraFactorList获取因子载荷
            rawCurrRiskFactorMatrix = self.getRiskFactorMatrix(barraFactorList, date)
            rawCurrIndustryFactorMatrix = self.getIndustryFactorMatrix(date)
            rawCurrMLFactorMatrix = self.getMLFactorMatrix(date, adjustedMLFactorDF)
            # 合并风格因子与行业因子矩阵，可能存在个股有行业因子载荷但没有风格因子载荷
            rawCurrRiskFactorMatrix.loc[
                rawCurrIndustryFactorMatrix.index, rawCurrIndustryFactorMatrix.columns] = rawCurrIndustryFactorMatrix

            rawCurrRiskFactorMatrix.loc[
                rawCurrMLFactorMatrix.columns, 'MLFactor'] = np.array(rawCurrMLFactorMatrix).reshape([np.shape(rawCurrMLFactorMatrix)[1],1])

            rawCurrOpenPriceReturn = self.OpenPriceReturn.loc[date, self.barraStock]

            # 删除当下时间缺省的个股的特征与收益
            riskFactorStockList = []
            openPriceStockList = []
            industryStockList = rawCurrIndustryFactorMatrix.index.tolist()
            MLFactorStockList = rawCurrMLFactorMatrix.columns.tolist()
            for stock in rawCurrRiskFactorMatrix.index:
                if np.isnan(sum(rawCurrRiskFactorMatrix.loc[stock, :])) == 0 and stock in industryStockList and stock in MLFactorStockList:
                    riskFactorStockList.append(stock)
            for stock in rawCurrOpenPriceReturn.index:
                if np.isnan(rawCurrOpenPriceReturn[stock]) == 0:
                    openPriceStockList.append(stock)
            finalStockList = [i for i in riskFactorStockList if i in openPriceStockList]
            currRiskFactorMatrix = rawCurrRiskFactorMatrix.loc[finalStockList, :]
            currOpenPriceReturn = rawCurrOpenPriceReturn[finalStockList]

            # 计算最终使用个股个数,为了防止用的股票数太少
            # stockNumDF.loc[date,'availableStockNum'] = len(finalStockList)

            # 当日因子收益率为K*1向量，K为因子数，1为一天收益
            [currFactorReturn, currCountryReturn, score] = self.getCrossSectionReturn(currRiskFactorMatrix,currOpenPriceReturn,ridgeparams)
            FactorReturn.loc[date, currRiskFactorMatrix.columns] = currFactorReturn
            FactorReturn.loc[date, 'country'] = currCountryReturn
            scoreDF.loc[date, 'score'] = score
        return [FactorReturn, scoreDF]