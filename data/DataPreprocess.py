# -*- coding: utf-8 -*-
"""
Created on Sum Feb 6 2022

@author: Wang
"""
from data.basicData import BasicData
import numpy as np
import pandas as pd
import pickle
rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'
basicInformation = pd.read_pickle(rootPath+'data\\describe_dict.pickle')
factorDF = pd.DataFrame()
for factor in basicInformation:
    factorDF.loc[factor,'nan_perc'] = basicInformation[factor]['nan_perc']
    factorDF.loc[factor, 'max'] = basicInformation[factor]['max']
    factorDF.loc[factor, 'min'] = basicInformation[factor]['min']
    factorDF.loc[factor, 'median'] = basicInformation[factor]['median']
    factorDF.loc[factor, 'percent0.75'] = basicInformation[factor]['perc0.75']
    factorDF.loc[factor, 'percent0.25'] = basicInformation[factor]['perc0.25']
factorDF.to_csv(rootPath+'data\\factorFeature.csv',encoding = 'utf_8_sig')