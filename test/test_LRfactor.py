""" 
@Time    : 2022/1/8 15:25
@Author  : Carl
@File    : test_MLfactor.py
@Software: PyCharm
"""
from data.basicData import BasicData
from LR_factor.Penal_linear import Penal_linearregre
from LR_factor.Average_linear import Average_linearregre
from LR_factor.basicfactor_LR import Basicfactor_LR
import os
print(os.path.abspath('.'))
# year='2011-2013'
# year='2014-2016'
# year='2017-2020'
years=['2011-2013','2014-2016','2017-2020']
#%%


# for year in years:
#     BasicData.load_data(year=year)
#     c=Basicfactor_LR()
#     c.load_data(year=year,ini=True,nums='all')

#%%
# for method_name in ['LASSO','ELAST','RIDGE']:
for method_name in ['PLS','PCR']:
    for year in years:
        # method_name='RIDGE'
        if method_name in ['PLS','PCR']:
            c = Average_linearregre(method=method_name,year=year)
        elif method_name in ['OLS','LASSO','RIDGE','ELAST']:
            c = Penal_linearregre(method=method_name,year=year)
        # c.data_preparation(ini=False,nums='all')
        model,predict_y = c.rolling_fit()


#%% 保存模型
# import pickle
# import os
# print(os.path.abspath('.'))
# with open('./ML_factor/result/{}_result_{}.pickle'.format(method_name,year), 'wb') as file:
#     pickle.dump(model, file)
# file.close()

#%% 
# for year in ['2011-2013','2014-2016','2017-2020','all']:
#     BasicData.load_data(year)