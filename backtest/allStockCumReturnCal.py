# -*- coding: utf-8 -*-
"""
Created on Mon Feb 8 22:02:51 2022

@author: Wang
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

openPriceInfo = pd.read_pickle(f'./data/pickleMaskingOpenPrice.pickle')
openPriceDF = openPriceInfo['openPrice']
openPriceReturnDF = pd.DataFrame(index = openPriceDF.index,columns=openPriceDF.columns)
# 计算收益率
for timeInd in range(1,len(openPriceDF.index.tolist())):
    lastTimeInd = timeInd-1
    openPriceReturnDF.iloc[timeInd,:] = openPriceDF.iloc[timeInd,:]/openPriceDF.iloc[lastTimeInd,:]-1
dailyReturnDF = pd.DataFrame(index = openPriceDF.index)
for timeInd in openPriceReturnDF.index:
    dailyReturnDF.loc[timeInd,'return'] = (0 if np.isnan(np.mean(openPriceReturnDF.loc[timeInd,:])) else np.mean(openPriceReturnDF.loc[timeInd,:]))
### 回测结果
cumReturnDF = (1+dailyReturnDF).cumprod()
performance = pd.DataFrame(index=[0],
                           columns=['cumRts(%)', 'annualVol(%)', 'maxDrawdown(%)', 'winRate(%)', 'SharpeRatio'])
performance['cumRts(%)'] = round(100 * (cumReturnDF.iloc[-1] - 1), 2).tolist()[0]
performance['annualRts(%)'] = round(100 * (cumReturnDF.iloc[-1] ** (252 / len(cumReturnDF)) - 1), 2).tolist()[0]
performance['annualVol(%)'] = round(100 * dailyReturnDF.std() * ((237 * 250) ** 0.5), 2).tolist()[0]
expandingMaxNetValue = cumReturnDF.expanding().max()
drawdown = cumReturnDF / expandingMaxNetValue - 1
performance['maxDrawdown(%)'] = round(-100 * drawdown.min(), 2).tolist()[0]
longShortRts = dailyReturnDF
performance['winRate(%)'] = round(100 * (longShortRts > 0).sum() / longShortRts.shape[0], 2).tolist()[0]
performance['SharpeRatio'] = round(longShortRts.mean(skipna=True) / longShortRts.std(), 4).tolist()[0]

# 打印累计收益率曲线
fig, ax1 = plt.subplots(1, 1, figsize=(16, 10))
ax1.plot(((1 + dailyReturnDF).cumprod() - 1) * 100, linewidth=3, label='long cum rts')
ax1.set_ylabel('cum returns(%)')
plt.show()

performance.to_csv(f'./backtest/allStockReturnBacktest.csv',encodings = 'utf_8_sig')