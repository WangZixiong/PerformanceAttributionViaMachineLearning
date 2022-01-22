# -*- coding: utf-8 -*-
"""
Created on Sun Jan  16 16:02:51 2020

@author: Wang
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import sqlalchemy as sa
eng = sa.create_engine('oracle+cx_oracle://student1901212644:student1901212644@219.223.208.52:1521/orcl')
rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\data\\'
# 获取全部A股代码
sqlQueryAllStock = 'SELECT S_INFO_WINDCODE FROM FILESYNC.AShareDescription'
AllStockCodesDF = pd.read_sql(sqlQueryAllStock,eng)
AllStockCodesList = []
for stockInd in AllStockCodesDF.index:
    stockCode = AllStockCodesDF.loc[stockInd,'s_info_windcode']
    if stockCode[0] == 'A':
        continue
    AllStockCodesList.append(stockCode)

# 获取日开盘价与复权因子
Return = pd.DataFrame()
for i in tqdm(range(int(np.ceil(len(AllStockCodesList)/1000)))):
    #语句每次只能获取1000只基金数据，需要分批放入AnnualReturn中
    tempStockList = AllStockCodesList[i*1000:min(i*1000+1000,len(AllStockCodesList))]
    tempStockListStr = '('
    for stock in tempStockList:
        tempStockListStr+='\''+stock+'\','
    tempStockListStr = tempStockListStr[:-1]+')'
    sqlQueryAdjustedReturn = 'SELECT TRADE_DT,S_INFO_WINDCODE,S_DQ_OPEN,S_DQ_ADJFACTOR FROM FILESYNC.AShareEODPrices WHERE TRADE_DT > 20100101'
    tempReturn = pd.read_sql(sqlQueryAdjustedReturn,eng)
    Return = pd.concat([Return,tempReturn])
Return.sort_values(by = 's_info_windcode', ascending = True,inplace = True)
Return.to_csv(rootPath+'2011-2020年个股开盘价与复权因子.csv',encoding = 'utf_8_sig')

# 整理为dateNum*stockNum的格式
dateList = list(set(Return['trade_dt']))
dateList.sort()
AllStockCodesList = list(set(Return['s_info_windcode']))
AllStockCodesList.sort()
openPriceDF = pd.DataFrame(index = dateList,columns = AllStockCodesList)
adjustedDF = pd.DataFrame(index = dateList,columns = AllStockCodesList)
for date in tqdm(dateList):
    currReturn = Return.loc[Return['trade_dt'] == date]
    # 尝试向量化操作，转化index后对应带入
    # currReturn.set_index(currReturn['s_info_windcode'])

    for stock in currReturn['s_info_windcode']:
        if stock in currReturn['s_info_windcode'].tolist():
            openPriceDF.loc[date, stock] = currReturn.loc[currReturn['s_info_windcode'] == stock]['s_dq_open'].values[0]
            adjustedDF.loc[date, stock] = currReturn.loc[currReturn['s_info_windcode'] == stock]['s_dq_adjfactor'].values[0]
