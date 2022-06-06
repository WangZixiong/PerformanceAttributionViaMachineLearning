""" 
@Time    : 2022/1/8 15:25
@Author  : Carl
@File    : test_MLfactor.py
@Software: PyCharm
"""
import pandas as pd
from tqdm import tqdm

from ML_factor.RandomForest import RF
import pickle
c = RF('open')
c.data_preparation()
# 滚动模型
factorExposure,R2oosDF,featureImportanceDF = c.rolling_fit()
pickleDcit = {}
pickleDcit['stk_loading'] = factorExposure
pickleDcit['r2'] = R2oosDF
pickleDcit['feature_importance'] = featureImportanceDF
with open('./factor/RF_stk_loading.pickle','wb') as file:
    pickle.dump(pickleDcit,file)
