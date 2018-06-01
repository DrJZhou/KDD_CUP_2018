# -*- coding: utf-8 -*-
import numpy as np
import sys
# import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import time
from sklearn.externals import joblib
import math
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASEPATH = '../dataset/tmp/'
OHE = OneHotEncoder(sparse=False)


def get_score(pred, valid_y_exp):
    return np.mean(np.abs(pred - valid_y_exp) / (pred + valid_y_exp) * 2)


def mape_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def scoring(reg, x, y):
    pred = reg.predict(x)
    return -mape_error(pred, y)


def get_static_day(data):
    ans = np.array([[] for i in range(data.shape[0])])
    for i in range(int(data.shape[1] / 24)):
        ans = np.hstack((ans, get_static(data[:, i * 24:(i + 1) * 24])))
    return ans


# 矩阵 统计信息
def get_static(data):
    mean_ = np.mean(data, axis=1)
    median_ = np.median(data, axis=1)
    max_ = np.max(data, axis=1)
    sum_ = np.sum(data, axis=1)
    min_ = np.min(data, axis=1)
    var_ = np.var(data, axis=1)
    std_ = np.std(data, axis=1)
    ans = np.hstack((mean_, median_, max_, sum_, min_, var_, std_))

    print ans.shape
    ans = ans.reshape(-1, 7)
    print ans.shape
    return ans


length = 24 * 3
params = {
    # 'objective': myObjective6,
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
    'nthread': 30
}


# 训练模型
def model_train(params, train_X, test_X, train_Y, test_Y, model_file):
    reg = xgb.XGBRegressor(**params)
    reg.fit(train_X, train_Y, eval_set=[(train_X, train_Y), (test_X, test_Y)], verbose=100,
            early_stopping_rounds=20)
    pred = reg.predict(test_X)
    joblib.dump(reg, model_file)
    valid_y_exp = test_Y
    print get_score(pred, valid_y_exp)


# 矩阵onehot
def onehot_mat(data, city):
    ans = []
    for i in range(data.shape[0]):
        if city == "bj":
            tmp = np.zeros(70)
            tmp[int(data[i, 0]) - 1] = 1
            tmp[4 + int(data[i, 1]) - 1] = 1
            tmp[39 + int(data[i, 2]) - 1] = 1
            tmp[46 + int(data[i, 3]) - 1] = 1
        else:
            tmp = np.zeros(60)
            tmp[int(data[i, 0]) - 1] = 1
            tmp[5 + int(data[i, 1]) - 1] = 1
            tmp[29 + int(data[i, 2]) - 1] = 1
            tmp[36 + int(data[i, 3]) - 1] = 1
        ans.append(tmp)
    ans = np.array(ans)
    return ans


# 跑模型
def run(data, attribution, city):
    start_time = time.time()
    if attribution == "PM25":
        if city == "bj":
            X = data[:, :-3]
            Y = data[:, -3]
        else:
            X = data[:, :-2]
            Y = data[:, -2]
        data_2 = get_static(X[:, 4: 4 + length])
        static_day = get_static_day(X[:, 4: 4 + length])
    elif attribution == "PM10":
        if city == "bj":
            X = data[:, :-3]
            Y = data[:, -2]
        else:
            X = data[:, :-2]
            Y = data[:, -1]
        data_2 = get_static(X[:, 4 + length: 4 + length * 2])
        static_day = get_static_day(X[:, 4: 4 + length])
    else:
        X = data[:, :-3]
        Y = data[:, -1]
        data_2 = get_static(X[:, 4 + length * 2: 4 + length * 3])
        static_day = get_static_day(X[:, 4: 4 + length])
    data_1 = np.hstack((onehot_mat(X[:, :4], city), X[:, 4:], data_2, static_day))
    X = data_1
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=11)
    print train_X.shape, test_X.shape, train_Y.shape, test_Y.shape
    model_file = BASEPATH + city + '_' + attribution + '_best.model'
    model_train(params, train_X, test_X, train_Y, test_Y, model_file)
    end_time = time.time()
    print end_time - start_time


# 值 onthot
def onehot(index, length):
    a = np.zeros(length)
    a[index] = 1
    return a


# 向量统计信息
def get_static_one_sample_day(data):
    ans = []
    for i in range(int(len(data) / 24)):
        ans += list(get_static_one_sample(data[i * 24: (i + 1) * 24]))
    return np.array(ans)


# 向量统计信息
def get_static_one_sample(data):
    return np.array([np.mean(data), np.median(data), np.max(data), np.sum(data), np.min(data), np.var(data),
                     np.std(data)])


# 模型预测
def model_predict(tmp, reg, city, stations, attribution):
    # print tmp[:4]
    one_hot_station_id = onehot(int(tmp[1] - 1), len(stations[city].keys()))
    if city == 'bj':
        one_hot_type_id = onehot(int(tmp[0] - 1), 4)
    else:
        one_hot_type_id = onehot(int(tmp[0] - 1), 5)
    one_hot_week = onehot(int(tmp[2]), 7)
    one_hot_hour = onehot(int(tmp[3]), 24)
    one_hot_all = np.hstack((one_hot_type_id, one_hot_station_id, one_hot_week, one_hot_hour))
    # print one_hot_all.shape
    if attribution == "PM25":
        static = get_static_one_sample(tmp[4:4 + length])
        static_day = get_static_one_sample_day(tmp[4:4 + length])
    elif attribution == "PM10":
        static = get_static_one_sample(tmp[4 + length: 4 + length * 2])
        static_day = get_static_one_sample_day(tmp[4 + length: 4 + length * 2])
    else:
        static = get_static_one_sample(tmp[4 + length * 2: 4 + length * 3])
        static_day = get_static_one_sample_day(tmp[4 + length * 2: 4 + length * 3])
    test_X = np.hstack((one_hot_all, tmp[4:], static, static_day))
    # print test_X
    pred = reg.predict([test_X])
    return pred[0]


def main(city):
    filename = BASEPATH + city + '_training_pre.csv'
    data = np.loadtxt(filename, delimiter=",")
    print data.shape
    attribution = "PM25"
    run(data, attribution, city)
    attribution = "PM10"
    run(data, attribution, city)
    if city == 'bj':
        attribution = "O3"
        run(data, attribution, city)


if __name__ == '__main__':
    main(city="bj")
    main(city='ld')
