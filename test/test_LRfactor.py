""" 
@Time    : 2022/1/8 15:25
@Author  : Carl
@File    : test_MLfactor.py
@Software: PyCharm
"""
from ML_factor.CNN import CNN
from ML_factor.linear_regre_v2 import linearregre
import os
print(os.path.abspath('.'))

c = linearregre(normalize=2)
c.data_preparation()
model,predict_y = c.rolling_fit()
# model


# c = CNN()
# c.data_preparation()
# model = c.rolling_fit()
# model