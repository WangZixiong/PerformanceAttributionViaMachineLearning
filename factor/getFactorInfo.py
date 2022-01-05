# getFactorInfo：整理因子为相同格式：因子名称，因子类型，因子计算方式，因子值
# 具体格式为dictionary，包括factorName，factorType，factorCal，factorMatrix四个keys
import numpy as np
import pandas as pd
import scipy.io as scio
import pickle