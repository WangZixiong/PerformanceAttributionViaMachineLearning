"""
@Author: Carl
@Time: 2022/2/4 16:49
@SoftWare: PyCharm
@File: test_KNN.py
@coding=gbk
"""
from ML_factor.KNN import KNN

knn = KNN()
res = knn.rolling_fit()
res
