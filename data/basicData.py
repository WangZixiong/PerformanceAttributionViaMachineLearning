""" 
@Time    : 2022/1/8 20:03
@Author  : Carl
@File    : basicData.py
@Software: PyCharm
"""
import pandas as pd
rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'
class BasicData:
    with open(rootPath+r'data\2011-2020年个股收盘价与复权因子.csv', 'rb') as f1:
        basicMkt = pd.read_csv(f1, index_col=0)
    with open(rootPath+r'data\pickleFactors_40factor_gtja191.pickle', 'rb') as f2:
        basicFactor = pd.read_pickle(f2)

    def __new__(cls):
        return cls
