import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
# 本文件用于读取行业数据并整理为0-1矩阵
rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'
allIndustryInfoDF = pd.read_excel(rootPath+'data\\A股行业纯数据.xlsx',sheet_name = '总表',index_col = 0,header = 1)
industrySectorDict = {'周期产业':('石油石化','煤炭','有色金属','钢铁','基础化工'),'基础设施与地产产业':('电力及公用事业','建筑','建材','房地产','交通运输'),
                         '金融产业':('银行','非银行金融','综合金融'),'科技产业':('电子','通信','计算机','传媒'),
                         '消费产业':('轻工制造','商贸零售','消费者服务','纺织服装','食品饮料','农林牧渔'),'医疗健康产业':('医药','nonce'),
                         '制造产业':('机械','电力设备及新能源','国防军工','汽车','家电')}
industryDict = {'石油石化':'CITICSPetroleumPetrochemical','煤炭':'CITICSCoal','有色金属':'CITICSNonFerrousMetal','钢铁':'CITICSSteel','基础化工':'CITICSBasicChemical',
                '电力及公用事业':'CITICSElectricity','建筑':'CITICSArchitecture','建材':'CITICSBuildingMaterial','房地产':'CITICSRealEstate','交通运输':'CITICSTransportation',
                '银行':'CITICSBank','非银行金融':'CITICSNonBankFinancial','综合金融':'CITICSNonBankFinancial','电子':'CITICSElectronic','通信':'CITICSCommunication',
               '计算机':'CITICSComputer','传媒':'CITICSMedia','轻工制造':'CITICSLightManufacture','商贸零售':'CITICSCommercialRetail','消费者服务':'CITICSConsumerService',
               '纺织服装':'CITICSTextileClothing','食品饮料':'CITICSFoodBeverage','农林牧渔':'CITICSAgriculture','医药':'CITICSMedicine','机械':'CITICSMachine',
               '电力设备及新能源':'CITICSNewEnergy','国防军工':'CITICSMilitary','汽车':'CITICSAutomobile','家电':'CITICSHomeAppliance','综合':'CITICSComposite'}

# 生成0-1矩阵
factorExposureDF = pd.DataFrame()
ind = 0
for time in tqdm(allIndustryInfoDF.index):
    currTimeStockIndustry = allIndustryInfoDF.loc[time,:]
    for stock in currTimeStockIndustry.index:
        factorExposureDF.loc[ind,['SecuCode','MarketTime']] = stock,time
        if currTimeStockIndustry[stock] in industryDict:
            industryCol = industryDict[currTimeStockIndustry[stock]]
            factorExposureDF.loc[ind,industryCol] = 1
            ind += 1