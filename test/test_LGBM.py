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
factorExposure = lgbm.rolling_fit()
factorExposure.to_csv(rootPath+'factor\\LGBM因子载荷矩阵.csv')
