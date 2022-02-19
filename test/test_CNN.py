"""
@Author: Carl
@Time: 2022/1/30 22:37
@SoftWare: PyCharm
@File: test_CNN.py
"""
from ML_factor.CNN import CNN

cnn = CNN()
res = cnn.rolling_fit()
res
