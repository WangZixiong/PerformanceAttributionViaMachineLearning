"""
@Author: Carl
@Time: 2022/2/19 11:10
@SoftWare: PyCharm
@File: cleanFactors_gtja191_20220204.py
"""
import pickle
import scipy.io as io
import numpy as np

rawData = io.loadmat('./data/rawData/calcFactors_gtja191_20220204_.mat')

factor_num = len([f for f in rawData.keys() if f.startswith('alpha')])
date_num = rawData.get('sharedInformation')[0][0][0].shape[0]
stock_num = rawData.get('sharedInformation')[0][0][1].shape[0]

cleanData = np.zeros([stock_num, date_num, factor_num])
k = 0
for key, value in rawData.items():
    if key.startswith('alpha'):
        cleanData[:, :, k] = value[0][0][0].T
        k += 1

cleanData[np.isnan(cleanData)] = 0
cleanData[np.isinf(cleanData)] = 0

with open('./data/cleanData/cleanFactors_gtja191_20220204.pkl', 'wb') as file:
    pickle.dump(cleanData, file)

