# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 09:38:35 2022

@author: gong1078899525
"""

import os
import sys
sys.path.append(os.path.abspath('..'))
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from data.basicData import BasicData
from LR_factor.basicfactor_LR import Basicfactor_LR
# model=pd.read_pickle('../ML_factor/result/PCR_result.pickle')
# model_df=[pd.DataFrame(np.nan,index=range(model['model_info']['batch_size']),columns=range(model[0]['coef'].shape[0]))]

class process_model(Basicfactor_LR):
    def __init__(self,model_result,year):
        print('开始初始化')
        super().__init__()
        super().load_data(year,ini=False,nums='all')
        print('数据初始化完成')
        self.year=year
        self.T=BasicData.T
        self.N=BasicData.N
        self.model_result=model_result
    
    def input_model(self,model_result):
        self.model_result=model_result
    
    def get_compo_factor(self):
        # model_df=[]
        # if BasicData.begin_index==0:
        ini_df=pd.DataFrame(index=range(self.model_result['model_info']['batch_size']),columns=range(self.model_result[0]['coef'].shape[0]))
        
        model_df=[ini_df]
        if BasicData.begin_index!=0:
            model_r2=[]
        else:
            ini_series=pd.DataFrame(index=range(self.model_result['model_info']['batch_size']),columns=['ic','r2','r2_oos'])
            model_r2=[ini_series]
        # else:
        #     model_df=[]
        for i in self.model_result.keys():
            if i !='model_info':
                for num in range(self.model_result['model_info']['lookback_num']):
                    coef_=self.model_result[i]['coef']
                    # print(self.model_result[i].keys())
                    r2_=pd.DataFrame([self.model_result[i]['ic'],self.model_result[i]['r2'],self.model_result[i]['r2_oos'].numpy()],index=['ic','r2','r2_oos']).T
                    model_df.append(pd.DataFrame(coef_).T)
                    model_r2.append(r2_)
        model_df=pd.concat(model_df)
        model_r2=pd.concat(model_r2)
        self.model_df=model_df
        self.model_r2=model_r2
    
    def get_stk_loading(self):
        self.Y = torch.Tensor(np.array(self.Y))
        x_slice = self.X
        stk_matrix=[]
        for i in range(x_slice.shape[0]):
            i_matrix=np.sum(x_slice[i,:x_slice.shape[1]//10*10,:]*self.model_df,axis=1)
            stk_matrix.append(i_matrix)
        stk_matrix=pd.concat(stk_matrix,axis=1)
        if BasicData.begin_index!=0:
            stk_matrix=stk_matrix.iloc[200:,:]
        stk_matrix.index=range(stk_matrix.shape[0])
        self.stk_matrix=stk_matrix
        return stk_matrix
        
    def get_model_r2(self):
        
        # model_r2=self.model_r2.reset_index()
        return self.model_r2
        
