#### 1.1.1 因子数据收集目标
##### 收集各类日频因子，将各类因子信息整合为统一格式，对因子信息进行预处理。以价量因子为主
#### 1.1.2 因子数据收集步骤拆解
##### A.整合因子信息为统一格式
##### getFactorInfo
##### 对于不同文件格式的因子集，整合为一个类似于struct的结构，
##### 属性包括因子名称，因子类型，因子计算方式，因子值格式为个股*时间的二维矩阵
##### B.对因子矩阵进行预处理
##### slicefactor
##### 根据股票池、时间等限制对factor matrix进行切片
##### preprocessing
##### 对于因子切片数据进行预处理
##### a.遮罩：剔除上市未满一年、ST、涨跌停的股票等
##### b.极值处理、中性化与标准化处理等
#### 1.1.3 单因子回测
#### 1.1。3.1 单因子回测意义
##### 确保机器学习模型能够做出增益，合成因子的表现比全部的单因子更好
#### 1.1.3.2 单因子分析
##### 计算R^2,IC等指标
##### 计算因子之间的correlation
##### 比较合成因子与单个因子的表现
##### 在计算IC时，注意因子的生成时间，即盘中or盘后计算的因子
##### 调整收益率的选择点，T+1open开始，T+2open结束