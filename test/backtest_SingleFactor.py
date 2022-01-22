# -*- coding: utf-8 -*-
"""
Created on Mon Jan  15 22:02:51 2020

@author: Wang
"""
import numpy as np
import pandas as pd
from factor.factorBacktest import SingleFactorBacktest
rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'
factorInformation = pd.read_pickle(rootPath+'\\data\\pickleFactors_new40_gtja191.pickle')
factorName = 'alpha12'
factorExposure = factorInformation[factorName]['factorMatrix']

# test = SingleFactorBacktest(factorName, factorExposure, price, tradePoint='open')
