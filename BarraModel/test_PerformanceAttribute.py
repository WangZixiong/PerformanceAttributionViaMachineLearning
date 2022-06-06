from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import pickle
import warnings
from BarraModel.FundPerformanceAttribute import FundPerformanceAttribute
warnings.filterwarnings('ignore')
rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'
allFundNAV = pd.read_excel(rootPath+'data\\私募基金净值数据\\私募基金初筛\\私募基金筛选.xlsx',index_col = 0)
barra_ML_Return = pd.read_csv(rootPath+'TimeSeriesModel\\factorReturnData\\Fama-French三因子模型（极简算法）日收益率（截至到20220331）.csv',index_col = 0)
barra_ML_Return = barra_ML_Return.iloc[:,1:]
MLFactorReturn = pd.read_excel(rootPath+'backtest\\newFactorReturn\\日频换仓收益率\\LGBM Factor日频换仓无费率多空收益率0501.xlsx',index_col = 0)

# 计算所有基金的超额收益在所有因子上的显著性
filePath = rootPath+'backtest\\newFactorReturn\\日频换仓收益率\\'
names = os.listdir(filePath)

# significanceDF用于统计在不同p值下显著的基金个数
significanceDF = pd.DataFrame(columns = ['p<0.1','p<0.2'])
for fileName in tqdm(names[2:]):
    # MLFactorReturn = pd.read_excel(filePath+'backtest\\newFactorReturn\\日频换仓收益率\\LGBM Factor日频换仓无费率多空收益率0501.xlsx',index_col = 0)
    MLFactorReturn = pd.read_excel(filePath+fileName,index_col = 0)
    allFundAlphaDF = pd.DataFrame(columns = allFundNAV.columns)
    allFundPvalueDF = pd.DataFrame(columns = ['Alpha','MLFactor'],index = allFundNAV.columns)
    allFundTvalueDF = pd.DataFrame(columns = ['Alpha','MLFactor'],index = allFundNAV.columns)
    for fund in tqdm(allFundNAV.columns.tolist()):
        currFundName = fund
        currFundNAV = allFundNAV.loc[:, fund]
        fundPA = FundPerformanceAttribute(MLFactorReturn, currFundNAV,0,fund)
        results,alphaDF = fundPA.getExcessReturn()
        if type(alphaDF) == pd.DataFrame:
            allFundPvalueDF.loc[fund,:] = results.pvalues
            allFundTvalueDF.loc[fund,:] = results.tvalues
            for time in alphaDF.index:
                allFundAlphaDF.loc[time,fund] = alphaDF.loc[time,'alpha']
            nonNANList = [i for i in allFundAlphaDF if np.isnan(allFundAlphaDF.loc[time,fund]) == 0]
            if len(nonNANList) < 50:
                print(fund+'基金回归所得alpha数目过少')

    p10List = [i for i in allFundPvalueDF.loc[:,'Alpha'] if i < 0.1]
    p20List = [i for i in allFundPvalueDF.loc[:, 'Alpha'] if i < 0.2]
    significanceDF.loc[fileName[:-9],['p<0.1','p<0.2']] = len(p10List)/len(allFundPvalueDF.index), len(p20List)/len(allFundPvalueDF.index)
    # allFundPvalueDF.to_csv(rootPath+'TimeSeriesModel\\FFModelResult\\fundPvalue.csv',encoding='utf_8_sig')
    # allFundTvalueDF.to_csv(rootPath+'TimeSeriesModel\\FFModelResult\\fundTvalue.csv',encoding='utf_8_sig')
    allFundPvalueDF.to_csv(rootPath+'backtest\\newFactorSignificance\\'+fileName[:-9]+'与基金Alpha回归Pvalue.csv',encoding='utf_8_sig')
    allFundTvalueDF.to_csv(rootPath+'backtest\\newFactorSignificance\\'+fileName[:-9]+'与基金Alpha回归Tvalue.csv',encoding='utf_8_sig')

# 选择在0.2显著度下与ML因子收益率回归显著的基金作为计算载荷的对象
selectedFundPvalueDF= pd.read_csv(rootPath+'TimeSeriesModel\\FFModelResult\\568fund\\LGBM多空因子回归显著性检验\\selectedFundPvalue.csv',index_col = 0)
MLFactorReturn = pd.read_excel(rootPath+'backtest\\newFactorReturn\\LGBM Factor日频换仓无费率多空收益率0501.xlsx',index_col = 0)
allFundAlphaDF = pd.read_excel(rootPath+'TimeSeriesModel\\FFModelResult\\568fund\\fundAlpha.xlsx',index_col = 0)

selectedFundList = [fund for fund in selectedFundPvalueDF.index if selectedFundPvalueDF.loc[fund,'pvalue']<0.2]
AllPeriodFundFactorExposureDict,AllPeriodFactorTValueDict,AllPeriodFactorPValueDict,AllPeriodScoreDict = {},{},{},{}
for fund in tqdm(selectedFundList):
    currFundAlpha = allFundAlphaDF.loc[:,fund]
    fundPA = FundPerformanceAttribute(MLFactorReturn, currFundAlpha, 0, fund)
    AllPeriodFundFactorExposure, AllPeriodFactorTValue, AllPeriodFactorPValue, AllPeriodScore = fundPA.getRollingFundExposure(
        '年度')
    if type(AllPeriodFundFactorExposure) != str:
        AllPeriodFundFactorExposureDict[fund] = AllPeriodFundFactorExposure
        AllPeriodFactorTValueDict[fund] = AllPeriodFactorTValue
        AllPeriodFactorPValueDict[fund] = AllPeriodFactorPValue
        AllPeriodScoreDict[fund] = AllPeriodScore

# 0421 以FF三因子模型为基础，计算基金的超额收益
allFundAlphaDF = pd.DataFrame(columns = allFundNAV.columns)
allFundPvalueDF = pd.DataFrame(columns = ['Alpha','MKT','SMB','HML'],index = allFundNAV.columns)
allFundTvalueDF = pd.DataFrame(columns = ['Alpha','MKT','SMB','HML'],index = allFundNAV.columns)
allFundScore = pd.DataFrame(columns = ['R2'],index = allFundNAV.columns)
for fund in tqdm(allFundNAV.columns.tolist()):
    currFundName = fund
    currFundNAV = allFundNAV.loc[:, fund]
    fundPA = FundPerformanceAttribute(barra_ML_Return, currFundNAV,0,fund)
    results,alphaDF = fundPA.getExcessReturn()
    if type(alphaDF) == pd.DataFrame:
        for time in alphaDF.index:
            allFundAlphaDF.loc[time,fund] = alphaDF.loc[time,'alpha']
        allFundPvalueDF.loc[fund,:] = results.pvalues
        allFundTvalueDF.loc[fund,:] = results.tvalues
        allFundScore.loc[fund,['R2']] = results.rsquared
        nonNANList = [i for i in allFundAlphaDF if np.isnan(allFundAlphaDF.loc[time,fund]) == 0]
        if len(nonNANList) < 50:
            print(fund+'基金回归所得alpha数目过少')
allFundAlphaDF.to_csv(rootPath+'TimeSeriesModel\\FFModelResult\\fundAlpha.csv',encoding='utf_8_sig')
allFundPvalueDF.to_csv(rootPath+'TimeSeriesModel\\FFModelResult\\fundPvalue.csv',encoding='utf_8_sig')
allFundTvalueDF.to_csv(rootPath+'TimeSeriesModel\\FFModelResult\\fundTvalue.csv',encoding='utf_8_sig')
allFundScore.to_csv(rootPath+'TimeSeriesModel\\FFModelResult\\fundScore.csv',encoding='utf_8_sig')

# 0520 计算FF模型+机器学习因子的新四因子模型对基金收益的解释能力R square，与上述板块进行对比
allFundAlphaDF = pd.DataFrame(columns = allFundNAV.columns)
allFundScore = pd.DataFrame(columns = ['FF_R2','FFandML_R2'],index = allFundNAV.columns)
FF_Return = pd.read_excel(rootPath+'TimeSeriesModel\\factorReturnData\\Fama-French三因子模型（极简算法）日收益率（截至到20220331）.xlsx',index_col = 0)
MLFactorReturn = pd.read_excel(rootPath+'backtest\\newFactorReturn\\日频换仓收益率\\Lasso Factor日频换仓无费率多空收益率0501.xlsx',index_col = 0)
FF_ML_FactorReturnList = [i for i in barra_ML_Return.index if i in MLFactorReturn.index]
FF_ML_FactorReturn_long = pd.DataFrame(MLFactorReturn.loc[FF_ML_FactorReturnList,:])
FF_ML_FactorReturn_long.loc[FF_ML_FactorReturnList,['MKT','SMB','HML']] = FF_Return.loc[FF_ML_FactorReturnList,['MKT','SMB','HML']]
MLFactorReturn_longshort = pd.read_excel(rootPath+'backtest\\newFactorReturn\\日频换仓收益率\\Lasso Factor日频换仓无费率多空收益率0501.xlsx',index_col = 0)
FF_ML_FactorReturn_longshort = pd.DataFrame(MLFactorReturn_longshort.loc[FF_ML_FactorReturnList,:])
FF_ML_FactorReturn_longshort.loc[FF_ML_FactorReturnList,['MKT','SMB','HML']] = FF_Return.loc[FF_ML_FactorReturnList,['MKT','SMB','HML']]

for fund in tqdm(allFundNAV.columns.tolist()):
    currFundName = fund
    currFundNAV = allFundNAV.loc[:, fund]
    fundPA = FundPerformanceAttribute(FF_ML_FactorReturn_long, currFundNAV,0,fund)
    results,alphaDF = fundPA.getExcessReturn()
    if type(alphaDF) == pd.DataFrame:
        allFundScore.loc[fund,['FFandML_long_R2']] = results.rsquared
    fundPA = FundPerformanceAttribute(FF_ML_FactorReturn_longshort, currFundNAV, 0, fund)
    results, alphaDF = fundPA.getExcessReturn()
    if type(alphaDF) == pd.DataFrame:
        allFundScore.loc[fund, ['FFandML_longshort_R2']] = results.rsquared
    fundPA = FundPerformanceAttribute(FF_Return, currFundNAV, 0, fund)
    results, alphaDF = fundPA.getExcessReturn()
    if type(alphaDF) == pd.DataFrame:
        allFundScore.loc[fund, ['FF_R2']] = results.rsquared


# 将基金的超额收益与我的因子收益率进行回归，看显著性
allFundPvalueDF = pd.read_csv(rootPath+'TimeSeriesModel\\FFModelResult\\568fund\\fundPvalue.csv',index_col = 0)
allFundAlphaDF = pd.read_excel(rootPath+'TimeSeriesModel\\FFModelResult\\568fund\\fundAlpha.xlsx',index_col = 0)

MLFactorReturn = pd.read_excel(rootPath+'backtest\\newFactorReturn\\LGBM Factor日频换仓无费率多空收益率0501.xlsx',index_col = 0)
selectedFundPvalueDF = pd.DataFrame()
selectedFundTvalueDF = pd.DataFrame()
selectedFundScore = pd.DataFrame()
for fund in tqdm(allFundNAV.columns.tolist()):
    currFundName = fund
    currFundNAV = allFundAlphaDF.loc[:, fund]
    fundPA = FundPerformanceAttribute(MLFactorReturn, currFundNAV, 0, fund)
    if type(fundPA.getExcessReturn()[1]) == pd.DataFrame:
        results, alphaDF = fundPA.getExcessReturn()
        selectedFundPvalueDF.loc[fund, 'pvalue'] = results.pvalues[1]
        selectedFundTvalueDF.loc[fund, 'tvalue'] = results.tvalues[1]
        selectedFundScore.loc[fund, 'R2'] = results.rsquared
selectedFundPvalueDF.to_csv(rootPath+'TimeSeriesModel\\FFModelResult\\568fund\\LGBM多空因子回归显著性检验\\selectedFundPvalue.csv',encoding='utf_8_sig')
selectedFundTvalueDF.to_csv(rootPath+'TimeSeriesModel\\FFModelResult\\568fund\\LGBM多空因子回归显著性检验\\selectedFundTvalue.csv',encoding='utf_8_sig')
selectedFundScore.to_csv(rootPath+'TimeSeriesModel\\FFModelResult\\568fund\\LGBM多空因子回归显著性检验\\selectedFundScore.csv',encoding='utf_8_sig')


barra_ML_Return = pd.read_pickle(rootPath+'BarraModel\\barra_ML_Return.pickle')
# 只取风格因子+机器学习因子部分做输入
barraFactorList = ['Size','Beta','Momentum','ResidualVolatility','NonlinearSize','Book2Price','Liquidity','EarningsYield','Growth','Leverage','MLFactor']
barra_ML_Return = barra_ML_Return.loc[:,barraFactorList]
allFundNAV = pd.read_excel(rootPath+'data\\股票多头-私募基金筛选.xlsx',index_col = 0)
AllPeriodFundFactorExposureDict,AllPeriodFactorTValueDict,AllPeriodFactorPValueDict,AllPeriodScoreDict = {},{},{},{}
# 所有的基金统一进行归因
for fund in tqdm(allFundNAV.columns):
    currFundName = fund
    currFundNAV = allFundNAV.loc[:,fund]
    fundPA = FundPerformanceAttribute(MLFactorReturn,currFundNAV, 0, fund)
    AllPeriodFundFactorExposure, AllPeriodFactorTValue, AllPeriodFactorPValue, AllPeriodScore = fundPA.getRollingFundExposure('年度')
    if type(AllPeriodFundFactorExposure) != str:
        AllPeriodFundFactorExposureDict[fund] = AllPeriodFundFactorExposure
        AllPeriodFactorTValueDict[fund] = AllPeriodFactorTValue
        AllPeriodFactorPValueDict[fund] = AllPeriodFactorPValue
        AllPeriodScoreDict[fund] = AllPeriodScore

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
    fundPA = FundPerformanceAttribute(MLFactorReturn,currFundNAV, 0, fund)
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
    fundPA_withoutML = FundPerformanceAttribute(barra_Return, currFundNAV, 0, fund)
    newAllPeriodFundFactorExposure, newAllPeriodFactorTValue, newAllPeriodFactorPValue, newAllPeriodScore = fundPA.getRollingFundExposure('年度')
subPath = 'TimeSeriesModel\\FFModelResult\\568fund\\LGBM多空因子滚动回归结果\\'
writer1 = pd.ExcelWriter(rootPath + subPath+ '私募基金业绩归因Exposure.xlsx', engine='openpyxl')
for fund in AllPeriodFundFactorExposureDict:
    AllPeriodFundFactorExposureDict[fund].to_excel(writer1,sheet_name=fund,encoding = 'utf_8_sig')
writer1.save()
writer1.close()

writer2 = pd.ExcelWriter(rootPath + subPath+ '私募基金业绩归因Score.xlsx', engine='openpyxl')
for fund in AllPeriodScoreDict:
    AllPeriodScoreDict[fund].to_excel(writer2,sheet_name=fund,encoding = 'utf_8_sig')
writer2.save()
writer2.close()

writer3 = pd.ExcelWriter(rootPath + subPath+ '私募基金业绩归因TValue.xlsx', engine='openpyxl')
for fund in AllPeriodFactorTValueDict:
    AllPeriodFactorTValueDict[fund].to_excel(writer3,sheet_name=fund,encoding = 'utf_8_sig')
writer3.save()
writer3.close()

writer4 = pd.ExcelWriter(rootPath + subPath+ '私募基金业绩归因PValue.xlsx', engine='openpyxl')
for fund in AllPeriodFactorPValueDict:
    AllPeriodFactorPValueDict[fund].to_excel(writer4,sheet_name=fund,encoding = 'utf_8_sig')
writer4.save()
writer4.close()
