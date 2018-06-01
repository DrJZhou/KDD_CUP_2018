# -*- coding: utf-8 -*-
import numpy as np
# import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import make_scorer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import time
import math


def mape_error(y_true, y_pred):
    return -np.sqrt(np.sum((y_pred - y_true) * (y_pred - y_true)) * 1.0 / y_true.shape[0])


def scoring(reg, x, y):
    pred = reg.predict(x)
    return mape_error(pred, y)


OHE = OneHotEncoder(sparse=False)
NOR = StandardScaler()

BASEPATH = '../dataset/tmp/'
filename = BASEPATH + 'training_pre.csv'
start_time = time.time()
data = np.loadtxt(filename, delimiter=",")
data_1 = OHE.fit_transform(data[:, :4])
print data_1.shape
print data.shape
X = data[:, :-3]
Y = data[:, -3]
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=11)
print train_X.shape, test_X.shape, train_Y.shape, test_Y.shape
params = {
    # 'objective': reg:linear,
    'max_depth': 5,
    # 'learning_rate':0.001,
    'learning_rate': 0.001,
    'n_estimators': 2000,
    'gamma': 0.0,
    'min_child_weight': 2,
    'max_delta_step': 0,
    'subsample': 0.9,
    'colsample_bytree': 0.6,
    'colsample_bylevel': 0.9,
    'base_score': 10,
    'seed': 1
}

'''
'max_depth': 10,
'learning_rate': 0.001,
# 'learning_rate': 0.02,
'n_estimators': 3000,
'gamma': 0.8,
'min_child_weight': 2,
'reg_alpha': 0.001,
'max_delta_step': 0,
'subsample': 0.8,
'colsample_bytree': 0.8,
'colsample_bylevel': 0.9,
'base_score': 10,
'seed': 1,
'nthread': 10
'''

param_test1 = {
    'max_depth': range(7, 12, 2),
    'min_child_weight': [2],
    'gamma': [i / 10.0 for i in range(8, 9)],
    'subsample': [i / 10.0 for i in range(6, 10)],
    'colsample_bytree': [i / 10.0 for i in range(6, 10)],
    'reg_alpha': [0, 0.001, 0.001],
    'learning_rate': [0.001, 0.01, 0.1],
    'n_estimators': [1000, 2000, 3000]
}

param_test3 = {
    'gamma': [i / 10.0 for i in range(0, 10)]
}

param_test4 = {
    'subsample': [i / 10.0 for i in range(6, 10)],
    'colsample_bytree': [i / 10.0 for i in range(6, 10)]
}

param_test6 = {
    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
}

param_test7 = {
    'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05]
}

param_test8 = {
    'learning_rate': [0.001, 0.002, 0.005, 0.01, 0.1],
    'n_estimators': [2000, 3000, 4000],
}

cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=2)
gsearch1 = GridSearchCV(
    estimator=xgb.XGBRegressor(learning_rate=0.001, n_estimators=3000, max_depth=10, min_child_weight=2,
                               reg_alpha=0.001, gamma=0.6, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1,
                               seed=27), param_grid=params, scoring=scoring, n_jobs=-1, cv=cv, verbose=6)
print 1
gsearch1.fit(train_X, train_Y)
print gsearch1.grid_scores_, gsearch1.best_score_
print gsearch1.best_params_
