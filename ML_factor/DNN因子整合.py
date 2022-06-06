import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'

#### 默认5个KNN数据在prediction文件夹中，分别较KNN1,KNN2,KNN3,KNN4,KNN5.pkl
for num in range(5):
    currFile = pd.read_pickle(rootPath+f'factor\\DNNFactor\\KNN{num+1}.pkl')
    stockNum = np.shape(currFile)[0]//10
    for ind in tqdm(range(stockNum)):
        currStockFactor = currFile[ind*10:(ind+1)*10,:]
        currFactor = np.concatenate([np.zeros(200),currStockFactor.flatten('F')])
        currFactor = np.concatenate([currFactor,np.zeros(10)])
        if ind == 0:
            allStockFactor = currFactor
        else:
            allStockFactor = np.vstack((allStockFactor,currFactor))

    with open(rootPath+f'factor\\DNNFactor\\KNN{num + 1}Factor.pkl', 'wb') as file:
        pickle.dump(allStockFactor.T, file)