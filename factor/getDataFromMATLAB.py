# 本函数用于将gtja191因子的mat格式数据集转化为pickle格式
import numpy as np
import pandas as pd
import scipy.io as scio
import pickle

rootPath = 'C:\\Users\\Lenovo\\Desktop\\'
gtja191FactorDict = scio.loadmat(rootPath+'calcFactors_gtja191_40_20220106_.mat')
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
        pickleDict[factorName] = {'factorCalculation':factorCal,'factorMatrix':result}
pickleDict['sharedInformation'] = {'axis1Time':axis1Time,'axis2Stock':stockList}
file = open(rootPath+'pickleFactors_gtja191_40.pickle','wb')
pickle.dump(pickleDict,file)
file.close()