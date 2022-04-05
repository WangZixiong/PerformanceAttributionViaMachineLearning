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

ML_factor = pd.read_pickle(rootPath + 'factor\\ELAST_stk_loading.pickle')
ML_factor = ML_factor['stk_loading']
ML_factor.drop(['index'], axis=1, inplace=True)
barraFactorList = ['Size','Beta','Momentum','ResidualVolatility','NonlinearSize','Book2Price','Liquidity','EarningsYield','Growth','Leverage']

c = CrossSectionalRegression(OpenPriceData,barraExposure,industryRawDF,ML_factor)
[FactorReturn,scoreDF] = c.getFactorReturn(barraFactorList)
with open(rootPath+'BarraModel\\barraReturn.pickle','wb') as file:
    pickle.dump(FactorReturn)