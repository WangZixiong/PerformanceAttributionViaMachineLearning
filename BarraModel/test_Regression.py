import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
import warnings
from BarraModel.CrossSectionalRegression import CrossSectionalRegression
warnings.filterwarnings('ignore')
rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'
OpenPriceData = pd.read_pickle(rootPath + 'data\\pickleMaskingOpenPrice2729Times.pickle')
barraExposure = pd.read_pickle(rootPath + 'BarraModel\\barra_data\\barra_new.pkl')
industryRawDF = pd.read_excel(rootPath + 'data\\A股行业纯数据.xlsx', sheet_name='总表',index_col= 0 ,header= 1)

# 机器学习因子输入的默认格式为三部分
# ML_factorDict['stk_loading'],ML_factorDict['axis1Time'],ML_factorDict['axis2Stock']
ML_factor = pd.read_pickle(rootPath + 'factor\\ELAST_stk_loading.pickle')
ML_factor = ML_factor['stk_loading']
ML_factor.drop(['index'], axis=1, inplace=True)
# 因为ML_factor没有时间维度，所以扩充一下时间
sharedInformation = pd.read_pickle(rootPath+'data\\sharedInformation.pkl')
axis1Time,axis2Stock = sharedInformation['axis1Time'],sharedInformation['axis2Stock']
ML_factorDict = {}
ML_factorDict['stk_loading'],ML_factorDict['axis1Time'],ML_factorDict['axis2Stock'] = ML_factor,axis1Time,axis2Stock

barraFactorList = ['Size','Beta','Momentum','ResidualVolatility','NonlinearSize','Book2Price','Liquidity','EarningsYield','Growth','Leverage']
c = CrossSectionalRegression(OpenPriceData,barraExposure,industryRawDF,ML_factorDict)
# # 对机器学习因子做正交化
# MLFactorRegressionResult,adjustedMLFactorDF = c.orthogonalize(barraFactorList)
# with open(rootPath+'BarraModel\\MLFactorRegressionResult.pickle','wb') as file:
#     pickle.dump(MLFactorRegressionResult,file)
# with open(rootPath + 'BarraModel\\adjustedMLFactorDF.pickle', 'wb') as file:
#         pickle.dump(adjustedMLFactorDF, file)
#
# # 获取barra模型各因子的因子收益率
# [FactorReturn,scoreDF] = c.getFactorReturn(barraFactorList)
# with open(rootPath+'BarraModel\\barraReturn.pickle','wb') as file:
#     pickle.dump(FactorReturn,file)

# 获取barra+MLFactor的因子收益率
standardAdjustedMLFactorDF = pd.read_pickle(rootPath + 'BarraModel\\standardAdjustedMLFactorDF.pickle')
[ALLFactorReturn,scoreDF] = c.getBarraPlusMLFactorReturn(standardAdjustedMLFactorDF,barraFactorList)
with open(rootPath+'BarraModel\\barra_ML_Return.pickle','wb') as file:
    pickle.dump(ALLFactorReturn,file)