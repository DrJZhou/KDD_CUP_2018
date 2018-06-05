# KDD_CUP_2018
KDD CUP 2018 Top3 解决方案
---
## 队名： 头号玩家@ICA@CortexLabs
## 队员： 周杰（华东师范大学）  蔡恒兴（CortexLabs,中山大学）

---
### 网页链接：https://biendata.com/competition/kdd_2018
### KDD CUP 2018 Top4 链接（转发，侵权请联系删除）： https://github.com/piupiuup/kdd2018/blob/master/.gitignore/code
### KDD CUP 初赛 Top1 链接（转发，侵权请联系删除）：https://github.com/ryancheunggit/kddcup2018-of-fresh-air
### KDD CUP 2017 Task2 Top2 链接：https://github.com/12190143/Black-Swan
----
## 环境要求
### Python2.7
- sklearn
- pandas
- numpy
- XGBoost
- LightGBM

## 说明
- baseline/  数据预处理和主要的基本模型
- dataset/  存放数据集和临时数据
- image/  画图分析输出
- output/ 结果输出保存

## 大体思路
1. 数据预处理（主要是缺失值处理，如果连续缺失少于三个则线性填补，否则用3*24个连续值预测下一个值的预训练模型（pre_train.py）填充），采用一天的滑动窗口来增加数据
2. 主要模型
   - 1） lightgbm为主要模型，每次预测一个值预测48次，ld和bj的5个预测值分别训练5个模型，所有站点一起训练
   - 2） ExtraTreeRegression 每次预测48个值，分5个模型预测5个指标，所有站点一起训练
   - 3） xgboost思路同lightgbm
   - 4）lightgbm 对特征数据进行log处理预测，其他类似
3. 主要特征
   - 1）用前21天数据预测后两天的值，包括原始值，max,min,median等统计量，同时包含天，周等为单位的统计量
   - 3）天气特征，主要使用网格数据，附近一个站点的数据，这里只用了温度，湿度和气压数据
   - 4）天气预报，通过自己抓取得到，见crawl_data.py文件以及官方给定api数据
   - 5）是否周末，是否工作日，是否工作日第一天，最后一天，是否放假第一天，是否放假最后一天
   - 6）初预测指标以外的特征，比如预测PM25时，加入PM10的特征，发现只加入最后3-4天的数据比较好
4. 模型结果融合
   - 1）同一个模型用不同的参数来训练
   - 2）同一个模型用不同的数据来训练，通过控制时间范围和数据缺失的多少来获得不同的训练数据
   - 3）对获得结果进行简单mean或者median以及加权求和
   

---
具体方案说明待后续完善。。。


## 联系方式
- 1633636038@qq.com
- caihx1288@qq.com
