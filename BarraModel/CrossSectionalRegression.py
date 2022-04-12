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
params = {
    'fit_intercept':True,
    'copy_X':True,
    'n_jobs':-1
}
class CrossSectionalRegression():
    def __init__(self,OpenPriceData,barraExposure,industryRawDF,ML_factor):
        # 读取OpenPrice
        self.OpenPrice = OpenPriceData['openPrice']
        # 读取barraExposure，barraExposure时间跨度是20050104-20200316，个股对OpenPrice中的个股几乎全覆盖（差25只）
        self.barraExposure = barraExposure
        # 读取中信行业分类数据
        self.industryRawDF = industryRawDF
        # 读取机器学习因子
        self.MLFactor = ML_factor

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

        self.barraStock = [i for i in stockList if i in BarraStockList and i in industryStockList]
        self.barraTime = [i for i in tradeTimeList if i in rawBarraTradeTimeList and i in industryTimeList]
        self.barraStock.sort()
        self.barraTime.sort()

    def getFactorReturn(self,barraFactorList):
        # 获取个股收益率,每日个股收益为T+2日开盘/T+1日开盘，作为T+1期收益
        self.OpenPriceReturn = self.OpenPrice.shift(-2) / self.OpenPrice.shift(-1)-1
        # 截面回归获取当日因子收益率
        # 都是个股收益率与因子载荷回归，第一种是得到因子收益率，第二种是得到个股权重矩阵
        FactorReturn,scoreDF,stockNumDF = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
        for date in tqdm(self.barraTime[:10]):
            # 根据barraFactorList获取因子载荷
            rawCurrRiskFactorMatrix = self.getRiskFactorMatrix(barraFactorList,date)
            rawCurrIndustryFactorMatrix = self.getIndustryFactorMatrix(date)
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
            [currFactorReturn,currCountryReturn,score] = self.getCrossSectionReturn(currRiskFactorMatrix,currOpenPriceReturn)
            FactorReturn.loc[date,barraFactorList] = currFactorReturn
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
        currStockIndustryBoolDF = pd.DataFrame()
        for stock in currStockIndustry.index:
            if currStockIndustry[stock] in industryDict:
                currIndustry = industryDict[currStockIndustry[stock]]
                currStockIndustryBoolDF.loc[stock,currIndustry] = 1
        currStockIndustryBoolDF = currStockIndustryBoolDF.fillna(0)
        return currStockIndustryBoolDF
    def preprocessingFactor(self):
        # 为了防止出现因子载荷的共线性，使得解不唯一，故需要做正交化
        self.orthogonalize()
        self.normalize()

    def switchBarraStockCode(self,SecuCode):
        if SecuCode[0] == '6':
            return SecuCode+'.SH'
        else:
            return SecuCode + '.SZ'
    def getCrossSectionReturn(self,currRiskFactorMatrix,currOpenPriceReturn):
        model = sklearn.linear_model.LinearRegression(params)
        reg = model.fit(currRiskFactorMatrix,currOpenPriceReturn)
        currFactorReturn,currCountryReturn,score = reg.coef_,reg.intercept_,reg.score(currRiskFactorMatrix,currOpenPriceReturn)
        allFactorReturn = list(currFactorReturn)
        return [allFactorReturn,currCountryReturn,score]