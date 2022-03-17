""" 
@Time    : 2022/1/8 15:25
@Author  : Carl
@File    : test_MLfactor.py
@Software: PyCharm
"""
import pandas as pd
from tqdm import tqdm

from ML_factor.RandomForest import RF
c = RF('open')
c.data_preparation()
# 滚动模型
factorExposure,R2oosDF,featureImportanceDF = RF.rolling_fit()

factorExposure.to_csv('./factor/RF因子载荷矩阵0318.csv')
R2oosDF.to_csv('./factor/RF因子R2oos误差0318.csv')
featureImportanceDF.to_csv('./factor/RF因子重要性0318.csv')
