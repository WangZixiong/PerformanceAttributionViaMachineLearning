"""
@Author: Wang
@Time: 2022/3/2 11:49
@SoftWare: PyCharm
@File: test_KNN.py
"""
from ML_factor.LGBM import LGBM
rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'

lgbm = LGBM('open')
lgbm.data_preparation()
factorExposure,R2oosDF,featureImportanceDF = lgbm.rolling_fit()
factorExposure.to_csv(rootPath+'factor\\LGBM因子载荷矩阵0317.csv')
R2oosDF.to_csv(rootPath+'factor\\LGBM因子R2oos误差0317.csv')
featureImportanceDF.to_csv(rootPath+'factor\\LGBM因子重要性0317.csv')