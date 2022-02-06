# -*- coding: utf-8 -*-
"""
Created on Sum Feb 6 2022

@author: Wang
"""
from data.basicData import BasicData
import numpy as np
import pandas as pd
import pickle

# step1 遮罩
# 剔除上市未满一年、ST、涨跌停的股票等

# step2 极值处理