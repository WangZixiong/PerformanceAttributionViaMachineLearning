""" 
@Time    : 2022/1/8 20:03
@Author  : Carl
@File    : basicData.py
@Software: PyCharm
"""
import pandas as pd
class BasicData:
    rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'
    with open(rootPath+r'data\2011-2020年个股开盘价与复权因子.csv', 'rb') as f1:
        basicMkt = pd.read_csv(f1, index_col=0)
        # basicMkt = pd.read_pickle(f1)
    with open(rootPath+r'data\seperateData\pickleFactors_01_50_gtja191.pickle', 'rb') as f2:
        basicFactor = pd.read_pickle(f2)

    def __new__(cls):
        return cls
