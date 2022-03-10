"""
@Author: Wang
@Time: 2022/3/10 11:49
@SoftWare: PyCharm
@File: test_XGB.py
"""
from ML_factor.XGBoost import XGB
rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'

xgb = XGB('open')
xgb.data_preparation()
factorExposure = xgb.rolling_fit()
factorExposure.to_csv(rootPath+'factor\\XGBoost因子载荷矩阵.csv')