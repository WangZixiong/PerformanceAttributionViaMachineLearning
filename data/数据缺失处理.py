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
        factor_dict['max']=narray[~np.isnan(narray)].max()
        factor_dict['min']=narray[~np.isnan(narray)].min()
        factor_dict['median'] = np.median(narray[~np.isnan(narray)])
        for perc_n in [0.75,0.5,0.25]:
            factor_dict['perc'+str(perc_n)]=np.percentile(narray[~np.isnan(narray)],perc_n*100)
        df_desc=pd.DataFrame(narray).describe()
        nan_stock_id=np.where(df_desc.loc['count']==0)
        factor_dict['nan_stock']={'数量':[nan_stock_id[0].shape[0],'编号按照顺序从0开始'],'id':nan_stock_id}
        describe_dict[key]=factor_dict
#%%

with open('describe_dict.pickle','wb') as f:
    pickle.dump(describe_dict,f)