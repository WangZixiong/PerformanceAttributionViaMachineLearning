# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 11:20:10 2022

@author: gong1078899525
"""

from data.basicData import BasicData
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

describe_dict={}
for key in tqdm(BasicData.basicFactor.keys()):
    if key != 'sharedInformation':
        narray=BasicData.basicFactor[key]['factorMatrix']
        factor_dict={}
        factor_dict['nan_perc']=narray[np.isnan(narray)].shape[0]/narray.size
        if factor_dict['nan_perc']<0.99:
            factor_dict['max']=narray[~np.isnan(narray)].max()
            factor_dict['min']=narray[~np.isnan(narray)].min()
            for perc_n in [0.99,0.98,0.97,0.04,0.03,0.02,0.01]:
                factor_dict['perc'+str(perc_n)]=np.percentile(narray[~np.isnan(narray)],perc_n*100)
        df_desc=pd.DataFrame(narray).describe()
        nan_stock_id=np.where(df_desc.loc['count']==0)
        factor_dict['nan_stock']={'数量':[nan_stock_id[0].shape[0],'编号按照顺序从0开始'],'id':nan_stock_id}
        describe_dict[key]=factor_dict
#%%

with open('describe_dict.pickle','wb') as f:
    pickle.dump(describe_dict,f)