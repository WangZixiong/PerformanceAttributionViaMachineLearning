""" 
@Time    : 2022/1/8 15:25
@Author  : Carl
@File    : test_MLfactor.py
@Software: PyCharm
"""
import pandas as pd
from tqdm import tqdm

from ML_factor.RandomForest import RF

rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'
c = RF('open')
c.data_preparation()
# 滚动模型
fit_result = c.rolling_fit()
factorExposure = fit_result[0]
R2oosDF = fit_result[1]
factorExposure.to_csv(rootPath+'factor\\RF因子载荷矩阵.csv')
R2oosDF.to_csv(rootPath+'factor\\RF因子R2oos误差.csv')
