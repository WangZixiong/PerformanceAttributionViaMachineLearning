# 本函数用于将gtja191因子的mat格式数据集转化为pickle格式
import numpy as np
import pandas as pd
import scipy.io as scio
import h5py
import pickle

rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'

# 转化gtja191因子
gtja191FactorDict = scio.loadmat(rootPath+'data\\calcFactors_gtja191_20220111_.mat')
pickleDict = {}
for factorName in gtja191FactorDict:
    if 'alpha' in factorName:
        result = gtja191FactorDict[factorName][0,0][0]
        description = gtja191FactorDict[factorName][0,0][1][0,0]
        factorName = description[0][0]
        factorCal = description[1][0]
        axis1Time = gtja191FactorDict['sharedInformation'][0,0][0]
        axis2Stock = gtja191FactorDict['sharedInformation'][0,0][1]
        stockList = []
        # 个股代码没有后缀，填上
        for stockInd in range(np.size(axis2Stock)):
            if str(axis2Stock[stockInd][0][0])[0] == '6':
                stockStr = str(axis2Stock[stockInd][0][0])+'.SH'
            else:
                stockStr = str(axis2Stock[stockInd][0][0])+'.SZ'
            stockList.append(stockStr)

        # 识别是否是空集，若是则舍去
        mapResult = result.reshape([np.shape(result)[0]*np.shape(result)[1],1])
        notNanRate = len(mapResult[~np.isnan(mapResult)])/(np.shape(result)[0]*np.shape(result)[1])
        print(factorName,notNanRate)
        if notNanRate >0.2:
            pickleDict[factorName] = {'factorCalculation': factorCal, 'factorMatrix': result}
pickleDict['sharedInformation'] = {'axis1Time':axis1Time,'axis2Stock':stockList}
file = open(rootPath+'pickleFactors_new40_gtja191.pickle','wb')
pickle.dump(pickleDict,file)
file.close()
with open(r'C:\Users\Lenovo\Desktop\pickleFactors_new40_gtja191.pickle','rb') as file:
    dict_get = pickle.load(file)
file.close()

# 转化marketInfo文件中的开盘价信息
marketInfoDict = h5py.File(rootPath+'data\\marketInfo_securities_china.mat','r')
pickleDict = {}
# 获取股票列表，由于股票代码是字符，在HDF5中存储方式为ASC码的形式，需要比较复杂的转换
stockList = []
stockCodeFileNameArray = marketInfoDict['aggregatedDataStruct']['stock']['description']['tickers']['windTicker'][0]
for ind in range(len(stockCodeFileNameArray)):
    currStockStrDataSet = marketInfoDict[stockCodeFileNameArray[ind]]
    currStockStr = ''
    for strInd in range(len(currStockStrDataSet)):
        currStockStr += chr(currStockStrDataSet[strInd][0])
    stockList.append(currStockStr)
# 获取时间列表，时间是数字，直接用.value读取即可
timeList = list(marketInfoDict['aggregatedDataStruct']['sharedInformation']['allDates'].value[0])
pickleDict['sharedInformation'] = {'axis1Time':timeList,'axis2Stock':stockList}
# 获取于收盘价交易的个股收益,不知为何读入数据时发生了转置，为了保证列为时间序列，行为个股截面，转置回来
openPrice = marketInfoDict['aggregatedDataStruct']['stock']['properties']['open'].value.T
pickleDict['openPrice'] = openPrice

with open(rootPath+'data\\pickleMarketOpenPrice.pickle','wb') as file:
    pickle.dump(pickleDict,file)
file.close()