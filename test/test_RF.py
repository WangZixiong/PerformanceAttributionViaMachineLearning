""" 
@Time    : 2022/1/8 15:25
@Author  : Carl
@File    : test_MLfactor.py
@Software: PyCharm
"""
import pandas as pd
from tqdm import tqdm

from ML_factor.RandomForest import RF

c = RF()
c.data_preparation()
# grid search 研究最佳超参数
model_MSE_DF = pd.DataFrame()
# 滚动模型
R2oosDF = c.rolling_fit()
