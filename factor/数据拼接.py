# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 09:38:35 2022

@author: gong1078899525
"""

import pandas as pd
import numpy as np
model=pd.read_pickle('./ML_factor/result/PCR_result.pickle')
model_df=[pd.DataFrame(np.nan,index=range(model['model_info']['batch_size']),columns=range(model[0]['coef'].shape[0]))]

for i in model.keys():
    if i !='model_info':
        for num in range(model['model_info']['lookback_num']):
            coef_=model[i]['coef']
            model_df.append(pd.DataFrame(coef_).T)
            
model_df=pd.concat(model_df)