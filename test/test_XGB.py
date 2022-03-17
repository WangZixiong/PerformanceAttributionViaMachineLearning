"""
@Author: Wang
@Time: 2022/3/10 11:49
@SoftWare: PyCharm
@File: test_XGB.py
"""
from ML_factor.XGBoost import XGB
rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'
# rootPath = 'C:\\Users\\Administrator\\Documents\\GitHub\\PerformanceAttributionViaMachineLearning\\'

xgb = XGB('open')
xgb.data_preparation()
factorExposure,R2oosDF,featureImportanceDF = xgb.rolling_fit()
factorExposure.to_csv(rootPath+'factor\\XGBoost因子载荷矩阵0317.csv')
R2oosDF.to_csv(rootPath+'factor\\XGBoost因子R2oos误差0317.csv')
featureImportanceDF.to_csv(rootPath+'factor\\XGBoost因子重要性0317.csv')