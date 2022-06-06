# -*- coding: utf-8 -*-
"""
Created on Wen Apr 15 18:02:51 2022
本文件用于验证机器学习合成因子的有效性，包括两部分
第一，验证机器学习合成因子相较于Fama-French三因子有alpha
第二，验证机器学习合成因子相较于国泰君安191IC排名前十的因子有超额收益
@author: Wang
"""
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'
currFactor = 'PCR'
currPredictPeriod = '日频'
strategyType = '多头策略'
ML_factorReturn = pd.read_excel(rootPath + 'backtest\\newFactorReturn\\PCR Factor日频换仓无费率多头收益率0501.xlsx',index_col = 0)
hedgeFundReturn = pd.read_excel(rootPath + 'data\\私募基金净值数据\\私募基金初筛\\私募基金筛选.xlsx',index_col=0)
factorModelReturn = pd.read_excel(rootPath+'TimeSeriesModel\\factorReturnData\\Fama-French三因子模型（极简算法）日收益率（截至到20220331）.xlsx',index_col=0)
# factorModelReturn = pd.read_excel(rootPath+'data\\top10ICFactor\\NewTop10\\top10因子多空收益率0501.xlsx',index_col=0)

# 对齐时间
selectedFactorReturnDF = pd.DataFrame()
for time in ML_factorReturn.index:
    if time in factorModelReturn.index:
        selectedFactorReturnDF.loc[time,['MLFactor','SMB','HML','MKT']] = ML_factorReturn.loc[time,0],factorModelReturn.loc[time,'SMB'],factorModelReturn.loc[time,'HML'], factorModelReturn.loc[time,'MKT']
# selectedFactorReturnDF = pd.DataFrame()
# for time in tqdm(hedgeFundReturn.index):
#     if time in factorModelReturn.index:
#         selectedFactorReturnDF.loc[time,['MLFactor']] = hedgeFundReturn.loc[time,0]
#         for factor in factorModelReturn.columns:
#             selectedFactorReturnDF.loc[time,factor] = factorModelReturn.loc[time,factor]
selectedFactorReturnDF.fillna(0,inplace = True)
# 时间序列回归，证明alpha显著
# 用statsmodels中的sm.OLS拟合，便于看每个变量的t值
FactorPeriodReturn = sm.add_constant(selectedFactorReturnDF.loc[:,['SMB','HML','MKT']])
# FactorPeriodReturn = sm.add_constant(selectedFactorReturnDF.iloc[:,1:])

model = sm.OLS(selectedFactorReturnDF.loc[:,'MLFactor'],FactorPeriodReturn)
results = model.fit()
print(results.summary2())
resultDF = pd.DataFrame()
for ind in results.pvalues.index:
    resultDF.loc[ind, 'pvalues'] = results.pvalues[ind]
    resultDF.loc[ind, 'params'] = results.params[ind]
resultDF.to_csv(rootPath+'TimeSeriesModel\\factorSignificanceTest\\'+currFactor+currPredictPeriod+strategyType+'显著性检验.csv',encoding = 'utf_8_sig')
