""" 
@Time    : 2022/1/8 15:25
@Author  : Carl
@File    : test_MLfactor.py
@Software: PyCharm
"""
# from ML_factor.CNN import CNN
from ML_factor.linear_regre_v2 import linearregre
import os
print(os.path.abspath('.'))

c = linearregre(normalize=2)
# c.data_preparation()
c.data_preparation(ini=False,nums='all')
# c.data_preparation(ini=False)
model,predict_y = c.rolling_fit()
# model


# c = CNN()
# c.data_preparation()
# model = c.rolling_fit()
# model
#%%
import pickle
import os

with open('./ML_factor/result/LR_model_result.pickle', 'wb') as file:
    pickle.dump(model, file)
file.close()
# pd.to_pickle('')