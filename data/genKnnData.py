import pandas as pd
import numpy as np

window = 5

# y
data = pd.read_pickle('./data/pickleMaskingOpenPrice2729Times.pickle')
data_pred = pd.DataFrame(data['openPrice'])
data_pred = np.log(data_pred) - np.log(data_pred.shift(window-1))
data_pred.dropna(how='all', inplace=True)
data_pred.fillna(0, inplace=True)

# x
x1 = pd.read_pickle('./data/newPickleFactors_2011-2014_gtja191.pickle')
# x1 = x1['sharedInformation']
x2 = pd.read_pickle('./data/newPickleFactors_2015-2018_gtja191.pickle')
# x2 = x2['sharedInformation']
x3 = pd.read_pickle('./data/newPickleFactors_2019-2022_gtja191.pickle')
# x3 = x3['sharedInformation']

x = dict()
for key in x1.keys():
    if key != 'sharedInformation':
        x[key] = np.vstack((x1[key].values, x2[key].values))
        x[key] = np.vstack((x[key], x3[key].values))
del x1
del x2
del x3

for i, key in enumerate(x.keys()):
    if i == 0:
        factor = x[key][:, :, None]
    else:
        factor = np.concatenate((factor, x[key][:, :, None]), axis=2)
del x
factor = factor[window-1:, :, :]
# x T * N * K np.narray
# y T * N np.narray

# 录入sharedInformation
x1 = pd.read_pickle('./data/newPickleFactors_2011-2014_gtja191.pickle')
x1 = x1['sharedInformation']
x2 = pd.read_pickle('./data/newPickleFactors_2015-2018_gtja191.pickle')
x2 = x2['sharedInformation']
x3 = pd.read_pickle('./data/newPickleFactors_2019-2022_gtja191.pickle')
x3 = x3['sharedInformation']

tickers = x1['axis2Stock']
date = x1['axis1Time'] + x2['axis1Time'] + x3['axis1Time']
date = date[window-1:]
data_pred = data_pred.loc[:, tickers].values
pd.to_pickle(data_pred, './data/cleanData/y_prediction.pkl')
pd.to_pickle(factor, './data/cleanData/x.pkl')
info = dict()
info['tickers'] = tickers
info['date'] = date
pd.to_pickle(info, './data/cleanData/dataInfo.pkl')

# load data
# x = pd.read_pickle('./data/cleanData/x.pkl')
# y = pd.read_pickle('./data/cleanData/y_prediction.pkl')
