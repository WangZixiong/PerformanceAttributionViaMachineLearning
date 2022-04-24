from tqdm import tqdm
import pandas as pd
import numpy as np

import pickle
import warnings
from BarraModel.FundPerformanceAttribute import FundPerformanceAttribute
warnings.filterwarnings('ignore')
rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'
allFundNAV = pd.read_excel(rootPath+'data\\股票多头-私募基金筛选(按日期排序).xlsx',index_col = 0)
barra_ML_Return = pd.read_excel(rootPath+'TimeSeriesModel\\factorReturnData\\Fama-French三因子模型（极简算法）日收益率（截至到20220331）.xlsx',index_col = 0)
barra_ML_Return = barra_ML_Return.iloc[:,1:]

# 0421 计算基金的超额收益
# allFundAlphaDF = pd.DataFrame(columns = allFundNAV.columns)
# allFundPvalueDF = pd.DataFrame(columns = ['Alpha','MKT','SMB','HML'],index = allFundNAV.columns)
# allFundTvalueDF = pd.DataFrame(columns = ['Alpha','MKT','SMB','HML'],index = allFundNAV.columns)
# allFundScore = pd.DataFrame(columns = ['R2'],index = allFundNAV.columns)
# for fund in tqdm(allFundNAV.columns):
#     currFundName = fund
#     currFundNAV = allFundNAV.loc[:, fund]
#     fundPA = FundPerformanceAttribute(barra_ML_Return, currFundNAV)
#     results,alphaDF = fundPA.getExcessReturn()
#     for time in alphaDF.index:
#         allFundAlphaDF.loc[time,fund] = alphaDF.loc[time,'alpha']
#     allFundPvalueDF.loc[fund,:] = results.pvalues
#     allFundTvalueDF.loc[fund,:] = results.tvalues
#     allFundScore.loc[fund,['R2']] = results.rsquared
# allFundAlphaDF.to_csv(rootPath+'TimeSeriesModel\\FFModelResult\\fundAlpha.csv',encoding='utf_8_sig')
# allFundPvalueDF.to_csv(rootPath+'TimeSeriesModel\\FFModelResult\\fundPvalue.csv',encoding='utf_8_sig')
# allFundTvalueDF.to_csv(rootPath+'TimeSeriesModel\\FFModelResult\\fundTvalue.csv',encoding='utf_8_sig')
# allFundScore.to_csv(rootPath+'TimeSeriesModel\\FFModelResult\\fundScore.csv',encoding='utf_8_sig')

# 将基金的超额收益与我的因子收益率进行回归，看显著性
allFundPvalueDF = pd.read_csv(rootPath+'TimeSeriesModel\\FFModelResult\\fundPvalue.csv',index_col = 0)
allFundAlphaDF = pd.read_excel(rootPath+'TimeSeriesModel\\FFModelResult\\fundAlpha.xlsx',index_col = 0)
analysisFundList = []
for fund in allFundPvalueDF.index:
    if allFundPvalueDF.loc[fund,'Alpha']<0.1:
        analysisFundList.append(fund)
analysisFundList = allFundNAV.columns.tolist()

MLFactorReturn = pd.read_excel(rootPath+'backtest\\newFactorReturn\\LGBM Factor日频换仓无费率多空收益率0422.xlsx',index_col = 0)
selectedFundPvalueDF = pd.DataFrame(index = analysisFundList)
selectedFundTvalueDF = pd.DataFrame(index = analysisFundList)
selectedFundScore = pd.DataFrame(index = analysisFundList)
for fund in tqdm(analysisFundList):
    currFundName = fund
    currFundNAV = allFundAlphaDF.loc[:, fund]
    fundPA = FundPerformanceAttribute(MLFactorReturn, currFundNAV, 2021, fund)
    if fundPA.getExcessReturn() != 0:
        results, alphaDF = fundPA.getExcessReturn()
        selectedFundPvalueDF.loc[fund, 'pvalue'] = results.pvalues[1]
        selectedFundTvalueDF.loc[fund, 'tvalue'] = results.tvalues[1]
        selectedFundScore.loc[fund, 'R2'] = results.rsquared


barra_ML_Return = pd.read_pickle(rootPath+'BarraModel\\barra_ML_Return.pickle')
# 只取风格因子+机器学习因子部分做输入
barraFactorList = ['Size','Beta','Momentum','ResidualVolatility','NonlinearSize','Book2Price','Liquidity','EarningsYield','Growth','Leverage','MLFactor']
barra_ML_Return = barra_ML_Return.loc[:,barraFactorList]
allFundNAV = pd.read_excel(rootPath+'data\\股票多头-私募基金筛选.xlsx',index_col = 0)
AllPeriodFundFactorExposureDict,AllPeriodFactorTValueDict,AllPeriodFactorPValueDict,AllPeriodScoreDict = {},{},{},{}

#对单一基金进行归因
fund = allFundNAV.columns.tolist()[0]
currFundName = fund
currFundNAV = allFundNAV.loc[:, fund]
fundPA = FundPerformanceAttribute(barra_ML_Return, currFundNAV)
AllPeriodFundFactorExposure, AllPeriodFactorTValue, AllPeriodFactorPValue, AllPeriodScore = fundPA.getRollingFundExposure(
    '年度')
barraWithourMLFactorList = ['Size', 'Beta', 'Momentum', 'ResidualVolatility', 'NonlinearSize', 'Book2Price',
                            'Liquidity', 'EarningsYield', 'Growth', 'Leverage']
barra_Return = barra_ML_Return.loc[:, barraWithourMLFactorList]
fundPA_withoutML = FundPerformanceAttribute(barra_Return, currFundNAV)
newAllPeriodFundFactorExposure, newAllPeriodFactorTValue, newAllPeriodFactorPValue, newAllPeriodScore = fundPA_withoutML.getRollingFundExposure(
    '年度')
AllPeriodScoreCompare = AllPeriodScore
AllPeriodScoreCompare.loc[:,'Score Without ML'] = np.array(newAllPeriodScore)

# 所有的基金统一进行归因
for fund in tqdm(allFundNAV.columns):
    currFundName = fund
    currFundNAV = allFundNAV.loc[:,fund]
    fundPA = FundPerformanceAttribute(barra_ML_Return,currFundNAV)
    AllPeriodFundFactorExposure, AllPeriodFactorTValue, AllPeriodFactorPValue, AllPeriodScore = fundPA.getRollingFundExposure('年度')
    if type(AllPeriodFundFactorExposure) != str:
        AllPeriodFundFactorExposureDict[fund] = AllPeriodFundFactorExposure
        AllPeriodFactorTValueDict[fund] = AllPeriodFactorTValue
        AllPeriodFactorPValueDict[fund] = AllPeriodFactorPValue
        AllPeriodScoreDict[fund] = AllPeriodScore
        # AllPeriodFundFactorExposure.to_csv(rootPath+'analysis\\'+fund+'业绩归因.csv',encoding = 'utf_8_sig')
        # 只取风格因子+机器学习因子部分做输入
    barraWithourMLFactorList = ['Size','Beta','Momentum','ResidualVolatility','NonlinearSize','Book2Price','Liquidity','EarningsYield','Growth','Leverage']
    barra_Return = barra_ML_Return.loc[:,barraWithourMLFactorList]
    fundPA_withoutML = FundPerformanceAttribute(barra_Return, currFundNAV)
    newAllPeriodFundFactorExposure, newAllPeriodFactorTValue, newAllPeriodFactorPValue, newAllPeriodScore = fundPA.getRollingFundExposure('年度')

writer1 = pd.ExcelWriter(rootPath + 'analysis\\私募基金业绩归因Exposure.xlsx', engine='openpyxl')
for fund in AllPeriodFundFactorExposureDict:
    AllPeriodFundFactorExposureDict[fund].to_excel(writer1,sheet_name=fund,encoding = 'utf_8_sig')
writer1.save()
writer1.close()

writer2 = pd.ExcelWriter(rootPath + 'analysis\\私募基金业绩归因Score.xlsx', engine='openpyxl')
for fund in AllPeriodScoreDict:
    AllPeriodScoreDict[fund].to_excel(writer2,sheet_name=fund,encoding = 'utf_8_sig')
writer2.save()
writer2.close()

writer3 = pd.ExcelWriter(rootPath + 'analysis\\私募基金业绩归因TValue.xlsx', engine='openpyxl')
for fund in AllPeriodFactorTValueDict:
    AllPeriodFactorTValueDict[fund].to_excel(writer3,sheet_name=fund,encoding = 'utf_8_sig')
writer3.save()
writer3.close()

writer4 = pd.ExcelWriter(rootPath + 'analysis\\私募基金业绩归因PValue.xlsx', engine='openpyxl')
for fund in AllPeriodFactorPValueDict:
    AllPeriodFactorPValueDict[fund].to_excel(writer4,sheet_name=fund,encoding = 'utf_8_sig')
writer4.save()
writer4.close()
