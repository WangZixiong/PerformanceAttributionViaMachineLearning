"""
@Author: Carl
@Time: 2022/1/29 20:45
@SoftWare: PyCharm
@File: get_cnnData.py
"""
import pickle
import pandas as pd
import numpy as np

data = pd.read_pickle('./data/rawData/pickleFactors_gtja191_211211.pickle')
look_back_num = 10

"""
matlab时间戳 转成 python日期的方法，例：734508
（734508-719529）* 86400
"""
#%%
# 去除nan
cleanData = np.zeros((data['alpha2']['factorMatrix'].shape[1], data['alpha2']['factorMatrix'].shape[0], len(data)-1))
for i, f in enumerate(data):
    if f != 'sharedInformation':
        cleanData[:, :, i] = data[f]['factorMatrix'].T
cleanData[np.isnan(cleanData)] = 0
cleanData[np.isinf(cleanData)] = 0
with open('./data/cleanData/cleanFactors_gtja191_211211.pkl', 'wb') as file:
    pickle.dump(cleanData, file)

#%%
# 整理CNN输入数据
with open('./data/cleanData/cleanFactors_gtja191_20220204.pkl', 'rb') as file:
    cleanData = pickle.load(file)
dates = pd.DataFrame(data['sharedInformation']['axis1Time'], columns=['timestamp'])
dates['timestamp'] = (dates['timestamp']-719529) * 86400
dates['date'] = pd.to_datetime(dates['timestamp'], unit='s')
dates['year'] = dates['date'].apply(lambda x: x.year)
groups = dates.groupby(dates.year).groups
groups[2011] = groups[2011][9:]
for g in groups:
    cnnData = np.zeros((cleanData.shape[0], len(groups[g]), look_back_num, cleanData.shape[2]))
    for i, d in enumerate(groups[g]):
        cnnData[:, i, :, :] = cleanData[:, d-9:d+1, :]
    with open(f'./data/cnnData/{g}_cnnFactors_gtja191_211211', 'wb') as file:
        pickle.dump(cnnData, file)

group_info = dict()
groupInfo = dict()
groupLimit = dict()
for key, value in groups.items():
    groupInfo.update(dict([(v, key) for v in list(value)]))

limit = 0
for key, value in groups.items():
    limit += len(value)
    groupLimit[key] = limit

group_info['groupInfo'] = groupInfo
group_info['groupLimit'] = groupLimit
with open(f'./data/cnnData/group_info.pkl', 'wb') as file:
    pickle.dump(group_info, file)

#%%
# labels
data = pd.read_pickle('./data/rawData/pickleMarketForwardOpenPrice.pickle')
data_pred = pd.DataFrame(data['openPrice'])
data_pred.loc[:, :] = np.log(data_pred.values) - np.log(data_pred.shift(1).values)
data_pred.fillna(0, inplace=True)
data_classify = data_pred.copy()
data_classify[data_classify > 0] = 1
data_classify[data_classify <= 0] = 0
data_classify = data_classify.astype(int)
with open('./data/cleanData/y_prediction.pkl', 'wb') as f1:
    pickle.dump(data_pred.values.T, f1)
with open('./data/cleanData/y_classification.pkl', 'wb') as f2:
    pickle.dump(data_classify.values.T, f2)
