# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 18:05:00 2022

@author: gong1078899525
"""

from data.basicData import BasicData
import numpy as np
import pandas as pd
import pickle
import os
rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'
with open(rootPath+'data\\pickleMaskingMarketInfo.pickle','rb') as f:
    market_info=pd.read_pickle(f)

# with open('./data/2011-2020年个股收盘价与复权因子.csv', 'rb') as f1:
#     basicMkt = pd.read_csv(f1, index_col=0)


# with open('./data/pickleFactors_40factor_gtja191.pickle', 'rb') as f2:
#     basicFactor = pd.read_pickle(f2)

#%%
process_price=market_info['ForwardOpenPrice'].copy()

#%%跌停涨停
process_price[market_info['ForwardLowPrice']==market_info['ForwardHighPrice']]=np.nan
# process_price[process_price==market_info['ForwardLowPrice']]=np.nan
#%%上市一年遮罩
def get_time(series):
    # if isinstance(series[0],int):
    py_series=(series-719529)*86400
    time_series=pd.to_datetime(py_series,unit='s')
    return time_series

    # else:
    #     return series


market_info['listDate']=get_time(market_info['listDate'][0])
market_info['sharedInformation']['axis1Time']=get_time(np.array(market_info['sharedInformation']['axis1Time']))
process_price=pd.DataFrame(process_price,index=market_info['sharedInformation']['axis1Time'],columns=market_info['sharedInformation']['axis2Stock'])

for i in range(market_info['ForwardOpenPrice'].shape[1]):
    list_date=market_info['listDate'][i]
    list_date_1=list_date+pd.Timedelta(days=365)
    process_price[(process_price.index<list_date_1) & (process_price.index<list_date)].iloc[:,i]=np.nan
    
#%%st去掉
process_price[market_info['stTable'].astype(bool)]=np.nan
pickleDict = {}
pickleDict['openPrice'] = process_price
with open(rootPath+'data\\pickleMaskingOpenPrice.pickle','wb') as file:
    pickle.dump(pickleDict,file)
file.close()