""" 
@Time    : 2022/1/8 15:25
@Author  : Carl
@File    : test_MLfactor.py
@Software: PyCharm
"""
from ML_factor.RandomForest import RF

c = RF()
c.data_preparation()
model = c.rolling_fit()
model
