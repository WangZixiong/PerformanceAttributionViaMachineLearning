""" 
@Time    : 2022/1/8 15:25
@Author  : Carl
@File    : test_MLfactor.py
@Software: PyCharm
"""
from LR_factor.model_process import process_model
import os
import pandas as pd
import numpy as np
import pickle
print(os.getcwd())
print(os.path.abspath('.'))
years=['2011-2013','2014-2016','2017-2020']
# years=['2017-2020']
# model_name=['ELAST','LASSO','RIDGE']
model_name=['LASSO','RIDGE','ELAST','PCR','PLS']
# model_name=['RIDGE','ELAST']
# model_name=['LASSO']
# c=process_model(0)
# c.data_preparation(ini=False,nums='all')
#%%
for model_n in model_name:
    stk_loading_list=[]
    r2_list=[]
    for year in years:
        model=pd.read_pickle('./ML_factor/result/{}_result_{}.pickle'.format(model_n,year))
        c=process_model(0,year)
        # model_list.append(model)
        c.input_model(model)
        c.get_compo_factor()
        stk_loading=c.get_stk_loading()
        r2_=c.get_model_r2()
        stk_loading_list.append(stk_loading)
        r2_list.append(r2_)
    stk_loading_list=pd.concat(stk_loading_list,axis=0).reset_index()
    r2_list=pd.concat(r2_list,axis=0).reset_index().drop(labels='index',axis=1)
    with open('./ML_factor/result/{}_stk_loading.pickle'.format(model_n),'wb') as file:
        pickle.dump({'stk_loading':stk_loading_list,'r2':r2_list}, file)
    file.close()
#%%
# model1=pd.read_pickle('./ML_factor/result/ELAST_stk_loading.pickle')
# model2=pd.read_pickle('./ML_factor/result/LASSO_stk_loading.pickle')
# model3=pd.read_pickle('./ML_factor/result/RIDGE_stk_loading.pickle')
