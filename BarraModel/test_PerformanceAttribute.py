from tqdm import tqdm
import pandas as pd
import pickle
import warnings
from BarraModel.FundPerformanceAttribute import FundPerformanceAttribute
# warnings.filterwarnings('ignore')
rootPath = 'C:\\Users\\Lenovo\\Desktop\\毕设材料\\PerformanceAttributionViaMachineLearning\\'
barra_ML_Return = pd.read_pickle(rootPath+'BarraModel\\barra_ML_Return.pickle')
allFundNAV = pd.read_excel(rootPath+'data\\股票多头-私募基金筛选.xlsx',index_col = 0)
for fund in allFundNAV.columns:
    currFundName = fund
    currFundNAV = allFundNAV.loc[:,fund]
    fundPA = FundPerformanceAttribute(barra_ML_Return,currFundNAV)
    AllPeriodFundFactorExposure, AllPeriodScore = fundPA.getRollingFundExposure('季度')
    if type(AllPeriodFundFactorExposure) != str:
        AllPeriodFundFactorExposure.to_csv(rootPath+'analysis\\'+fund+'业绩归因.csv',encoding = 'utf_8_sig')
        # 0416请无比搞清楚score到底是干嘛的，如何验证每个因子的有效性
        AllPeriodScore.to_csv(rootPath+'analysis\\'+fund+'Score.csv',encoding = 'utf_8_sig')
