""" 
@Time    : 2022/1/8 15:25
@Author  : Carl
@File    : test_MLfactor.py
@Software: PyCharm
"""
import pandas as pd
from tqdm import tqdm

from ML_factor.RandomForest import RF

c = RF()
c.data_preparation()
# grid search 研究最佳超参数
model_MSE_DF = pd.DataFrame()
#for MaxDepth in tqdm([2,4,6]):
#    for N_Estimators in tqdm([100,200,300]):
#        currMSE = c.whole_sample_fit(MaxDepth,N_Estimators)
#        model_MSE_DF.loc[MaxDepth,N_Estimators] = currMSE
# 滚动模型
R2oosDF = c.rolling_fit()
