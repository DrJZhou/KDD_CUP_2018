# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import time
import re
from sklearn.externals import joblib
import requests
import sys
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR, NuSVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import make_scorer
from data_processing import *
import xgboost as xgb
import math
import lightgbm as lgb
import cPickle as pickle
from weather_data_processing import *
import xlrd

reload(sys)
sys.setdefaultencoding('utf-8')
base_path_1 = "../dataset/"
base_path_2 = "../dataset/tmp/"
base_path_3 = "../output/"

base_log = 1


station_id_change = {
    'miyunshuiku_aq': 'miyunshuik_aq',
    'wanshouxigong_aq': 'wanshouxig_aq',
    'nongzhanguan_aq': 'nongzhangu_aq',
    'xizhimenbei_aq': 'xizhimenbe_aq',
    'fengtaihuayuan_aq': 'fengtaihua_aq',
    'aotizhongxin_aq': 'aotizhongx_aq',
    'yongdingmennei_aq': 'yongdingme_aq'
}


def get_nongli(city):
    pass


def get_train_test_data(city, length=24 * (3 * 7 + 2)):
    weather_attr_need = ['temperature', 'pressure', 'humidity']
    if city == 'bj':
        attr_need = ["PM25_Concentration", "PM10_Concentration", "O3_Concentration", 'time_year',
                     'time_month', 'time_day', 'time_week', 'time_hour']
    else:
        attr_need = ["PM25_Concentration", "PM10_Concentration", 'time_year', 'time_month',
                     'time_day', 'time_week', 'time_hour']
    stations = load_station()
    ans_history = history_data(city=city, stations=stations, start_day="2017-08-01", end_day="2018-04-10")
    weather_history = history_weather_data(city=city, start_day="2017-08-01", end_day="2018-04-10")
    ans_current = get_all_processing_data(city, start_day="2018-04-11", end_day="2018-04-29")
    weather_current = load_all_weather_data(city, start_day="2018-04-11", end_day="2018-04-29", crawl_data=False)
    weather_data = pd.concat([weather_history, weather_current["2018-04-11":"2018-04-29"]], axis=0)
    weather_groups = weather_data.groupby("station_id")
    # print ans_history
    num = 0
    ans = []
    for station, group in ans_history.items():
        if station == "zhiwuyuan_aq":
            continue
        grid_station = nearst[city][station]
        weather_data = weather_groups.get_group(grid_station).sort_index()
        station_num_id = stations[city][station]["station_num_id"]
        station_type_id = stations[city][station]["type_id"]
        if ans_current.has_key(station):
            group = pd.concat([ans_history[station], ans_current[station]["2018-04-11":"2018-04-29"]],
                              axis=0).sort_index()
        else:
            weather_data = weather_data[:"2018-04-10"]
        group = group.drop_duplicates()
        values = group[attr_need].values
        weather_values = weather_data[weather_attr_need].values
        print values.shape, weather_values.shape
        values = np.hstack([values, weather_values])
        for i in range(0, values.shape[0] - length + 1, 24):
            tmp = [station_num_id, station_type_id]
            if city == "bj":
                tmp += list(values[i + length - 24, 3: 7])
                values2 = values[i: i + length - 48, :3]
                values2 = list(values2.T.flatten())
                tmp += values2
                values2 = values[i: i + length, -3:]
                values2 = list(values2.T.flatten())
                tmp += values2
                values2 = values[i + length - 48: i + length, :3]
                values2 = list(values2.T.flatten())
                tmp += values2
            else:
                tmp += list(values[i + length - 24, 2: 6])
                values2 = values[i: i + length - 48, :2]
                values2 = list(values2.T.flatten())
                tmp += values2
                values2 = values[i: i + length, -3:]
                values2 = list(values2.T.flatten())
                tmp += values2
                values2 = values[i + length - 48:i + length, :2]
                values2 = list(values2.T.flatten())
                tmp += values2
            # print tmp
            tmp = np.array(tmp)
            if np.isnan(tmp).sum() > 0:
                num += 1
                continue
            ans.append(tmp)
    print num
    ans_history = history_data(city=city, stations=stations, start_day="2017-02-01", end_day="2017-05-31")
    weather_history = history_weather_data(city=city, start_day="2017-02-01", end_day="2017-05-31")
    weather_groups = weather_history.groupby("station_id")
    for station, group in ans_history.items():
        if station == "zhiwuyuan_aq":
            continue
        grid_station = nearst[city][station]
        weather_data = weather_groups.get_group(grid_station).sort_index()
        station_num_id = stations[city][station]["station_num_id"]
        station_type_id = stations[city][station]["type_id"]
        group = group.drop_duplicates()
        values = group[attr_need].values
        weather_values = weather_data[weather_attr_need].values
        values = np.hstack([values, weather_values])
        for i in range(0, values.shape[0] - length + 1, 24):
            tmp = [station_num_id, station_type_id]
            if city == "bj":
                tmp += list(values[i + length - 24, 3: 7])
                values2 = values[i: i + length - 48, :3]
                values2 = list(values2.T.flatten())
                tmp += values2
                values2 = values[i: i + length, -3:]
                values2 = list(values2.T.flatten())
                tmp += values2
                values2 = values[i + length - 48: i + length, :3]
                values2 = list(values2.T.flatten())
                tmp += values2
            else:
                tmp += list(values[i + length - 24, 2: 6])
                values2 = values[i: i + length - 48, :2]
                values2 = list(values2.T.flatten())
                tmp += values2
                values2 = values[i: i + length, -3:]
                values2 = list(values2.T.flatten())
                tmp += values2
                values2 = values[i + length - 48:i + length, :2]
                values2 = list(values2.T.flatten())
                tmp += values2
            # print tmp
            tmp = np.array(tmp)
            if np.isnan(tmp).sum() > 0:
                num += 1
                continue
            ans.append(tmp)
    print num
    ans = np.array(ans)
    np.savetxt(base_path_2 + city + '_training_weather_0201-0531_0801-0429.csv', ans, delimiter=',')


def mape_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def scoring(reg, x, y):
    pred = reg.predict(x)
    return -mape_error(pred, y)


def get_onehot_feature(data, city):
    ans = []
    for i in range(data.shape[0]):
        if city == "bj":
            tmp = np.zeros(53)
            tmp[int(data[i, 0]) - 1] = 1
            tmp[35 + int(data[i, 1]) - 1] = 1
            tmp[39 + (int(data[i, 5]) + 7 - 1) % 7] = 1
            tmp[46 + int(data[i, 5])] = 1
        else:
            tmp = np.zeros(43)
            tmp[int(data[i, 0]) - 1] = 1
            tmp[24 + int(data[i, 1]) - 1] = 1
            tmp[29 + (int(data[i, 5]) + 7 - 1) % 7] = 1
            tmp[36 + int(data[i, 5])] = 1
        ans.append(tmp)
    ans = np.array(ans)
    return ans


def get_statistic_feature(data):
    mean_ = np.mean(data, axis=1)
    median_ = np.median(data, axis=1)
    max_ = np.max(data, axis=1)
    sum_ = np.sum(data, axis=1)
    min_ = np.min(data, axis=1)
    var_ = np.var(data, axis=1)
    std_ = np.std(data, axis=1)
    ans = np.hstack((mean_, median_, max_, sum_, min_, var_, std_))
    ans = ans.reshape(-1, 7)
    return ans


def get_every_weekday_static(data):
    ans = []
    for i in range(data.shape[0]):
        tmp = data[i, :].reshape(-1, 24)
        mean_ = np.mean(tmp, axis=0)
        median_ = np.median(data, axis=0)
        max_ = np.max(data, axis=0)
        sum_ = np.sum(data, axis=0)
        min_ = np.min(data, axis=0)
        var_ = np.var(data, axis=0)
        std_ = np.std(data, axis=0)
        ans.append(np.hstack((mean_, median_, max_, sum_, min_, var_, std_)))
    return np.array(ans)


def get_all_statistic_feature_1(data):
    all_static = get_statistic_feature(data)
    every_weekday_static = get_every_weekday_static(data)
    day_static = np.array([[] for i in range(data.shape[0])])
    for i in range(14, int(data.shape[1] / 24)):
        day_static = np.hstack((day_static, get_statistic_feature(data[:, i * 24:(i + 1) * 24])))
    week_static = np.array([[] for i in range(data.shape[0])])
    for i in range(int(data.shape[1] / (7 * 24))):
        week_static = np.hstack((week_static, get_statistic_feature(data[:, i * 24:(i + 1) * 24])))
    ans = np.hstack((all_static, every_weekday_static, week_static, day_static))
    return ans


def get_all_statistic_feature(data):
    all_static = get_statistic_feature(data)
    every_weekday_static = get_every_weekday_static(data)
    day_static = np.array([[] for i in range(data.shape[0])])
    for i in range(14, int(data.shape[1] / 24)):
        day_static = np.hstack((day_static, get_statistic_feature(data[:, i * 24:(i + 1) * 24])))
    week_static = np.array([[] for i in range(data.shape[0])])
    for i in range(int(data.shape[1] / (7 * 24))):
        week_static = np.hstack((week_static, get_statistic_feature(data[:, i * (24 * 7):(i + 1) * (24 * 7)])))
    ans = np.hstack((all_static, every_weekday_static, week_static, day_static))
    # return all_static
    return ans


def get_holiday_weekend(data):
    holiday = ['2017-01-01', '2017-01-02', '2017-01-27', '2017-01-28', '2017-01-29',
               '2017-01-30', '2017-01-31', '2017-02-01', '2017-02-02', '2017-04-02',
               '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
               '2017-05-28', '2017-05-29', '2017-05-30', '2017-10-01', '2017-10-02',
               '2017-10-03', '2017-10-04', '2017-10-05', '2017-10-06', '2017-10-07',
               '2017-10-08', '2017-12-30', '2017-12-31',
               '2018-01-01', '2018-02-15', '2018-02-16', '2018-02-17', '2018-02-18',
               '2018-02-19', '2018-02-20', '2018-02-21', '2018-04-05', '2018-04-06',
               '2018-04-07', '2018-04-29', '2018-04-30', '2018-05-01', '2018-06-16',
               '2018-06-17', '2018-06-18']

    work = ['2017-01-22', '2017-02-04', '2017-04-01', '2017-05-27', '2017-09-30',
            '2018-02-11', '2018-02-24', '2018-04-08', '2018-04-28']
    rest_first_day = ['2017-01-27', '2017-02-05', '2017-04-02', '2017-05-28', '2017-10-01', '2018-02-15', '2018-02-25',
                      '2018-04-05', '2018-04-29']
    rest_last_day = ['2017-01-02', '2017-01-21', '2017-02-02', '2017-02-05', '2017-04-04', '2017-05-01', '2017-05-30',
                     '2018-01-01', '2018-02-21', '2018-04-07', '2018-05-01']
    work_firt_day = ['2017-01-03', '2017-01-22', '2017-02-03', '2017-04-05', '2017-05-02', '2017-05-31', '2018-01-02',
                     '2018-02-11', '2018-02-22', '2018-04-08', '2018-05-02']
    work_last_day = ['2017-01-26', '2017-02-04', '2017-04-01', '2017-05-27', '2017-09-30', '2018-02-14', '2018-02-24',
                     '2018-04-04', '2018-04-28']

    not_rest_first_day = ['2017-01-28', '2017-02-04', '2017-04-01', '2017-05-27', '2017-09-30', '2017-10-07',
                          '2018-02-17',
                          '2018-02-24', '2018-04-07', '2018-04-28']
    not_rest_last_day = ['2017-01-01', '2017-01-22', '2017-02-29', '2017-04-02', '2017-04-30', '2017-05-28',
                         '2017-10-01'
                         '2017-12-31', '2018-02-11', '2018-02-18', '2018-04-08', '2018-04-29']
    not_work_firt_day = ['2017-01-02', '2017-01-23', '2017-01-30', '2017-04-03', '2017-05-01', '2017-05-29',
                         '2017-10-02',
                         '2018-01-01', '2018-02-12', '2018-02-19', '2018-04-09', '2018-04-30']
    not_work_last_day = ['2017-01-27', '2017-02-03', '2017-03-31', '2017-05-26', '2017-09-29', '2017-10-06',
                         '2018-02-16',
                         '2018-02-23', '2018-04-04', '2018-04-06', '2018-04-27']
    holiday_flag = {}
    for day in holiday:
        holiday_flag[day] = 1
    work_flag = {}
    for day in work:
        work_flag[day] = 1
    ans = []
    for i in range(data.shape[0]):
        year = int(data[i, 0])
        month = int(data[i, 1])
        day = int(data[i, 2])
        tmp = []
        day_now = datetime(year, month, day) - timedelta(days=22)
        for j in range(23):
            day = day_now + timedelta(days=j)
            day_str = datetime_toString(day)
            # 判断是否周末
            week = day.weekday()
            if week >= 5:
                tmp.append(1)
            else:
                tmp.append(0)
            # 判断是否节假日
            if holiday_flag.has_key(day_str):
                tmp.append(1)
            else:
                tmp.append(0)
            # 判断是否工作
            if (week >= 5 and (not work_flag.has_key(day_str))) or holiday_flag.has_key(day_str):
                tmp.append(0)
            else:
                tmp.append(1)
            # 判断放假最后一天
            if (week == 6 and (not day_str in not_rest_last_day)) or (day_str in rest_last_day):
                tmp.append(1)
            else:
                tmp.append(0)
            # 判断上班第一天
            if (week == 0 and (not day_str in not_work_firt_day)) or (day_str in work_firt_day):
                tmp.append(1)
            else:
                tmp.append(0)
            # 判断上班最后一天
            if (week == 4 and (not day_str in not_work_last_day)) or (day_str in work_last_day):
                tmp.append(1)
            else:
                tmp.append(0)
            # 判断放假第一天
            if (week == 5 and (not day_str in not_rest_first_day)) or (day_str in rest_first_day):
                tmp.append(1)
            else:
                tmp.append(0)
        ans.append(tmp)
    return np.array(ans)


def get_all_feature_1(data, city, attr="PM25", length=3 * 7 * 24):
    onehot_feature = get_onehot_feature(data[:, :6], city)
    if city == "bj":
        holiday_weekend_feature = get_holiday_weekend(data[:, 2:6])
    if attr == "PM25":
        static_feature = get_all_statistic_feature_1(data[:, 6: 6 + length])
    elif attr == "PM10":
        static_feature = get_all_statistic_feature_1(data[:, 6 + length: 6 + length * 2])
    else:
        static_feature = get_all_statistic_feature_1(data[:, 6 + length * 2: 6 + length * 3])
    if city == "bj":
        all_feature = np.hstack((data[:, 6:], holiday_weekend_feature, onehot_feature, static_feature))
    else:
        all_feature = np.hstack((data[:, 6:], onehot_feature, static_feature))
    return all_feature
    pass


def get_all_feature(data, city, attr="PM25", length=3 * 7 * 24):
    onehot_feature = get_onehot_feature(data[:, :6], city)
    if city == 'bj':
        holiday_weekend_feature = get_holiday_weekend(data[:, 2:6])
    if attr == "PM25":
        static_feature = get_all_statistic_feature(data[:, 6: 6 + length])
        orign_data = np.hstack([data[:, 6 + 14 * 24: 6 + length], data[:, 6 + length + 18 * 24:6 + length * 2]])
    elif attr == "PM10":
        static_feature = get_all_statistic_feature(data[:, 6 + length: 6 + length * 2])
        orign_data = np.hstack([data[:, 6 + 18 * 24: 6 + length], data[:, 6 + length + 14 * 24:6 + length * 2]])
    else:
        static_feature = get_all_statistic_feature(data[:, 6 + length * 2: 6 + length * 3])
        orign_data = data[:, 6 + length * 2 + 14 * 24:6 + length * 3]
    # weather_feature = np.hstack(
    #     [data[:, 6 + length * 3 + 7 * 24: 6 + length * 4], data[:, 6 + length * 4 + 7 * 24: 6 + length * 5],
    #      data[:, 6 + length * 4 + 7 * 24: 6 + length * 5]])
    orign_data = np.log1p(orign_data)
    static_feature = np.log1p(static_feature)
    if city == "bj":
        all_feature = np.hstack(
            (orign_data, data[:, 6 + length * 3:], holiday_weekend_feature, onehot_feature, static_feature))
        # all_feature = np.hstack(
        #     (orign_data, weather_feature, holiday_weekend_feature, onehot_feature, static_feature))
    else:
        all_feature = np.hstack((orign_data, data[:, 6 + length * 3:], onehot_feature, static_feature))
        # all_feature = np.hstack((orign_data, weather_feature, onehot_feature, static_feature))
    return all_feature
    pass


def change_X_Y(X, Y):
    ans_X = []
    ans_Y = []
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            tmp = np.zeros(48)
            tmp[j] = 1
            ans_X.append(np.hstack((X[i, :], tmp)))
            ans_Y.append(Y[i, j])
    return np.array(ans_X), np.array(ans_Y)


def load_train_test(city, attr, type="0301-0531_0801-0410", load_from_feature_file=False):
    filename = base_path_2 + city + '_training_weather_' + type + '.csv'
    data = np.loadtxt(filename, delimiter=",")
    if city == "bj":
        X_origin = data[:, :-48 * 3]
    else:
        X_origin = data[:, :-48 * 2]
    if load_from_feature_file == False:
        X = get_all_feature(X_origin, city, attr=attr)
        np.savetxt(base_path_2 + city + "_" + attr + '_training_log_feature_X_weather_' + type + '_1.csv', X, delimiter=',')
    else:
        X = np.loadtxt(base_path_2 + city + "_" + attr + '_training_log_feature_X_weather_' + type + '_1.csv',
                       delimiter=",")
    if attr == "PM25":
        if city == "bj":
            Y = data[:, -48 * 3:-48 * 2]
        else:
            Y = data[:, -48 * 2:-48]
    elif attr == "PM10":
        if city == "bj":
            Y = data[:, -48 * 2:-48]
        else:
            Y = data[:, -48:]
    else:
        Y = data[:, -48:]
    # if type == "0301-0531_0801-0410":
    #     ans_X, ans_Y = change_X_Y(np.concatenate([X[:10500, :], X[-10500:, :]]),
    #                               np.concatenate([Y[:10500, :], Y[-10500:, :]]))
    # else:
    #     ans_X, ans_Y = change_X_Y(np.concatenate([X[:, :], X[:, :]]), np.concatenate([Y[:, :], Y[:, :]]))
    ans_X, ans_Y = change_X_Y(X, Y)
    ans_Y = np.log1p(ans_Y)
    train_X, test_X, train_Y, test_Y = train_test_split(ans_X, ans_Y, test_size=0.15, random_state=11)
    print train_X.shape, test_X.shape, train_Y.shape, test_Y.shape
    return train_X, test_X, train_Y, test_Y


def get_score(pred, valid_y_exp):
    return np.mean(np.abs(pred - valid_y_exp) / (pred + valid_y_exp) * 2)


model_param = {'lr': 0.01, 'depth': 10, 'tree': 1000, 'leaf': 400, 'sample': 0.9, 'seed': 3}
params = {
    # 'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'metric': {'l2', 'l1'},
    # 'max_depth': model_param['depth'],
    # 'num_leaves': model_param['leaf'],
    # 'min_data_in_leaf': 20,
    # 'learning_rate': model_param['lr'],
    # 'feature_fraction': 1,
    # 'bagging_fraction': model_param['sample'],
    # 'bagging_freq': 1,
    # 'bagging_seed': model_param['seed'],
    'verbose': 0
}

model_param1 = {'lr': 0.05, 'depth': 10, 'tree': 1500, 'leaf': 400, 'sample': 0.9, 'seed': 3}
params1 = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'metric': {'l2', 'l1'},
    'max_depth': model_param1['depth'],
    'num_leaves': model_param1['leaf'],
    'min_data_in_leaf': 20,
    'learning_rate': model_param1['lr'],
    'feature_fraction': 1,
    'bagging_fraction': model_param1['sample'],
    'bagging_freq': 1,
    'bagging_seed': model_param1['seed'],
    'verbose': 0
}

model_param2 = {'lr': 0.1, 'depth': 10, 'tree': 1000, 'leaf': 200, 'sample': 0.9, 'seed': 3}
params2 = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'metric': {'l2', 'l1'},
    'max_depth': model_param2['depth'],
    'num_leaves': model_param2['leaf'],
    'min_data_in_leaf': 20,
    'learning_rate': model_param2['lr'],
    'feature_fraction': 1,
    'bagging_fraction': model_param2['sample'],
    'bagging_freq': 1,
    'bagging_seed': model_param2['seed'],
    'verbose': 0
}

model_param3 = {'lr': 0.05, 'depth': 20, 'tree': 2000, 'leaf': 500, 'sample': 0.9, 'seed': 3}
params3 = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'metric': {'l2', 'l1'},
    'max_depth': model_param3['depth'],
    'num_leaves': model_param3['leaf'],
    'min_data_in_leaf': 20,
    'learning_rate': model_param3['lr'],
    'feature_fraction': 1,
    'bagging_fraction': model_param3['sample'],
    'bagging_freq': 1,
    'bagging_seed': model_param3['seed'],
    'verbose': 0
}


model_param4 = {'lr': 0.005, 'depth': 15, 'tree': 3000, 'leaf': 500, 'sample': 0.9, 'seed': 3}
params4 = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'metric': {'l2', 'l1'},
    'max_depth': model_param4['depth'],
    'num_leaves': model_param4['leaf'],
    'min_data_in_leaf': 20,
    'learning_rate': model_param4['lr'],
    'feature_fraction': 1,
    'bagging_fraction': model_param4['sample'],
    'bagging_freq': 1,
    'bagging_seed': model_param4['seed'],
    'verbose': 0
}


model_param5 = {'lr': 0.01, 'depth': 15, 'tree': 6000, 'leaf': 500, 'sample': 0.9, 'seed': 3}
params5 = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'metric': {'l2', 'l1'},
    'max_depth': model_param5['depth'],
    'num_leaves': model_param5['leaf'],
    'min_data_in_leaf': 20,
    'learning_rate': model_param5['lr'],
    'feature_fraction': 1,
    'bagging_fraction': model_param5['sample'],
    'bagging_freq': 1,
    'bagging_seed': model_param5['seed'],
    'verbose': 0
}


def smape_error(preds, train_data):
    labels = train_data.get_label()
    return 'error', np.mean(np.fabs(preds - labels) / (preds + labels) * 2), False

cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=2)


# CLF = GridSearchCV(
#     estimator=xgb.XGBRegressor(learning_rate=0.001, n_estimators=3000, max_depth=10, min_child_weight=2,
#                                reg_alpha=0.001, gamma=0.6, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1,
#                                seed=27), param_grid=params, scoring=scoring, n_jobs=-1, cv=cv, verbose=6)

def cal_best_params(city, attr, type="0301-0531_0801-0410", load_from_feature_file=False):
    train_X, test_X, train_Y, test_Y = load_train_test(city=city, attr=attr, type=type,
                                                       load_from_feature_file=load_from_feature_file)
    lgb_train = lgb.Dataset(train_X, train_Y)
    lgb_eval = lgb.Dataset(test_X, test_Y, reference=lgb_train)
    best_params = {}
    print("调参1：提高准确率")
    min_merror = float('Inf')
    for num_leaves in range(20, 450, 100):
        for max_depth in range(5, 16, 5):
            params['num_leaves'] = num_leaves
            params['max_depth'] = max_depth
            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=42,
                nfold=5,
                early_stopping_rounds=10,
                verbose_eval=True
            )
            mean_merror = pd.Series(cv_results['l2-mean']).min()
            if mean_merror < min_merror:
                min_merror = mean_merror
                best_params['num_leaves'] = num_leaves
                best_params['max_depth'] = max_depth
    params['num_leaves'] = best_params['num_leaves']
    params['max_depth'] = best_params['max_depth']


    print(params)
    return params


def train(city, attr, best_params1="1", type="0301-0531_0801-0410", load_from_feature_file=False):
    train_X, test_X, train_Y, test_Y = load_train_test(city=city, attr=attr, type=type,
                                                       load_from_feature_file=load_from_feature_file)
    # create dataset for lightgbm
    lgb_train = lgb.Dataset(train_X, train_Y)
    lgb_eval = lgb.Dataset(test_X, test_Y, reference=lgb_train)
    if best_params1 == "1":
        gbm = lgb.train(params1,
                        lgb_train,
                        num_boost_round=model_param1['tree'],
                        valid_sets=lgb_eval,
                        feval=smape_error,
                        early_stopping_rounds=20)
    elif best_params1 == "2":
        gbm = lgb.train(params2,
                        lgb_train,
                        num_boost_round=model_param2['tree'],
                        valid_sets=lgb_eval,
                        feval=smape_error,
                        early_stopping_rounds=20)
    elif best_params1 == "3":
        gbm = lgb.train(params3,
                        lgb_train,
                        num_boost_round=model_param3['tree'],
                        valid_sets=lgb_eval,
                        feval=smape_error,
                        early_stopping_rounds=20)
    elif best_params1 == "4":
        gbm = lgb.train(params4,
                        lgb_train,
                        num_boost_round=model_param4['tree'],
                        valid_sets=lgb_eval,
                        feval=smape_error,
                        early_stopping_rounds=20)
    else:
        gbm = lgb.train(params5,
                        lgb_train,
                        num_boost_round=model_param5['tree'],
                        feval=smape_error,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=20)
    test_Y1 = gbm.predict(test_X, num_iteration=gbm.best_iteration)
    score = get_score(np.expm1(test_Y1), np.expm1(test_Y))
    model_file = base_path_2 + city + '_' + attr + '_best_lightgbm_with_weather_log_params_' + best_params1 + '_' + type + '_1.model'
    # gbm.save_model(model_file)
    with open(model_file, 'wb') as fout:
        pickle.dump(gbm, fout)
    return score


def change_feature(feature):
    ans = []
    for i in range(48):
        tmp = np.zeros(48)
        tmp[i] = 1
        ans.append(np.hstack((feature, np.array(tmp))))
    return np.array(ans)


def predict(city, length=24 * (3 * 7 + 2), start_day="2018-04-11", end_day="2018-04-11", type="0301-0531_0801-0410",
            feature_first=False, best_params="1", caiyun = False):
    weather_attr_need = ['temperature', 'pressure', 'humidity']
    if city == 'bj':
        attr_need = ["PM25_Concentration", "PM10_Concentration", "O3_Concentration", 'time_year',
                     'time_month', 'time_day', 'time_week', 'time_hour']
    else:
        attr_need = ["PM25_Concentration", "PM10_Concentration", 'time_year', 'time_month',
                     'time_day', 'time_week', 'time_hour']
    stations = load_station()
    ans_history = history_data(city=city, stations=stations, start_day="2018-01-01", end_day="2018-04-10")
    ans_current = get_all_processing_data(city, start_day=start_day, end_day=end_day)
    weather_history = history_weather_data(city=city, start_day="2018-01-01", end_day="2018-04-10")
    weather_current = load_all_weather_data(city, start_day=start_day, end_day=end_day, crawl_data=False, caiyun=caiyun)
    weather_data = pd.concat([weather_history, weather_current], axis=0)
    weather_data = weather_data.drop_duplicates(["station_id", 'time'])
    weather_groups = weather_data.groupby("station_id")
    if feature_first == False:
        model_PM25_file = base_path_2 + city + '_PM25_best_lightgbm_with_weather_log_params_' + best_params + '_' + type + '_1.model'
        model_PM10_file = base_path_2 + city + '_PM10_best_lightgbm_with_weather_log_params_' + best_params + '_' + type + '_1.model'
    else:
        model_PM25_file = base_path_2 + city + '_PM25_best_lightgbm_with_weather_log_params_' + best_params + '_' + type + '.model'
        model_PM10_file = base_path_2 + city + '_PM10_best_lightgbm_with_weather_log_params_' + best_params + '_' + type + '.model'

    model_PM25 = pickle.load(open(model_PM25_file, 'rb'))
    model_PM10 = pickle.load(open(model_PM10_file, 'rb'))
    if city == "bj":
        if feature_first == False:
            model_O3_file = base_path_2 + city + '_O3_best_lightgbm_with_weather_log_params_' + best_params + '_' + type + '_1.model'
        else:
            model_O3_file = base_path_2 + city + '_O3_best_lightgbm_with_weather_log_params_' + best_params + '_' + type + '.model'
        model_O3 = pickle.load(open(model_O3_file, 'rb'))
    ans = ""
    for station in ans_history.keys():
        if city == "ld":
            if stations[city][station]["predict"] == False:
                continue
        # group1 = ans_history[station]
        # group2 = ans_current[station]["2018-04-11":]
        station_num_id = stations[city][station]["station_num_id"]
        station_type_id = stations[city][station]["type_id"]

        group = pd.concat([ans_history[station], ans_current[station]["2018-04-11":]], axis=0).sort_index()
        # print group
        grid_station = nearst[city][station]
        weather_data_tmp = weather_groups.get_group(grid_station).sort_index()
        values = group[attr_need].values
        weather_values = weather_data_tmp[weather_attr_need].values
        values = np.hstack([values, weather_values])
        # print values.shape
        i = values.shape[0] - length
        tmp = [station_num_id, station_type_id]
        if city == "bj":
            tmp += list(values[i + length - 24, 3: 7])
            values2 = values[i: i + length - 48, :3]
            values2 = list(values2.T.flatten())
            tmp += values2
            values2 = values[i: i + length, -3:]
            values2 = list(values2.T.flatten())
            tmp += values2
        else:
            tmp += list(values[i + length - 24, 2: 6])
            values2 = values[i: i + length - 48, :2]
            values2 = list(values2.T.flatten())
            tmp += values2
            values2 = values[i: i + length, -3:]
            values2 = list(values2.T.flatten())
            tmp += values2
        # print tmp.shape
        tmp = np.array(tmp)
        # print tmp.shape
        if feature_first == False:
            PM25_feature = get_all_feature(np.array([tmp]), city, attr="PM25")
            PM10_feature = get_all_feature(np.array([tmp]), city, attr="PM10")
        else:
            PM25_feature = get_all_feature_1(np.array([tmp]), city, attr="PM25")
            PM10_feature = get_all_feature_1(np.array([tmp]), city, attr="PM10")
        pred_PM25 = model_PM25.predict(change_feature(PM25_feature[0]), model_PM25.best_iteration)
        pred_PM10 = model_PM10.predict(change_feature(PM10_feature[0]), model_PM10.best_iteration)
        pred_PM25 = np.expm1(pred_PM25)
        pred_PM10 = np.expm1(pred_PM10)
        if station_id_change.has_key(station):
            station = station_id_change[station]
        if city == "bj":
            if feature_first == False:
                O3_feature = get_all_feature(np.array([tmp]), city, attr="O3")
            else:
                O3_feature = get_all_feature_1(np.array([tmp]), city, attr="O3")
            pred_O3 = model_O3.predict(change_feature(O3_feature[0]), model_O3.best_iteration)
            pred_O3 = np.expm1(pred_O3)
            for i in range(48):
                ans += station + "#" + str(i) + "," + str(pred_PM25[i]) + "," + str(pred_PM10[i]) + "," + str(
                    pred_O3[i]) + "\n"
                # print tmp.shape
        else:
            for i in range(48):
                ans += station + "#" + str(i) + "," + str(pred_PM25[i]) + "," + str(pred_PM10[i]) + ",0.0\n"
    return ans


def get_ans(type_, feature_first=False, start_day="2018-04-27", end_day="2018-04-29", best_params="1", caiyun = False):
    ans1 = predict(city='bj', start_day=start_day, end_day=end_day, type=type_, feature_first=feature_first,
                   best_params=best_params, caiyun=caiyun)
    ans2 = predict(city='ld', start_day=start_day, end_day=end_day, type=type_, feature_first=feature_first,
                   best_params=best_params, caiyun=caiyun)
    ans = "test_id,PM2.5,PM10,O3\n"
    if caiyun == True:
        ans_file = base_path_3 + end_day + "-lightgbm_weather_log_params_" + best_params + "_" + type_ + "_" + str(
            feature_first) + "_caiyun.csv"
    else:
        ans_file = base_path_3 + end_day + "-lightgbm_weather_log_params_" + best_params + "_" + type_ + "_" + str(
            feature_first) + ".csv"
    f_to = open(ans_file, 'wb')
    f_to.write(ans + ans1 + ans2)
    f_to.close()


def get_test(type, feature_first=False, best_params="1"):
    print type
    for i in range(11, 28):
        start_day = "2018-04-" + str(i)
        end_day = "2018-04-" + str(i)
        ans1 = predict(city='bj', start_day=start_day, end_day=end_day, type=type, feature_first=feature_first,
                       best_params=best_params)
        ans2 = predict(city='ld', start_day=start_day, end_day=end_day, type=type, feature_first=feature_first,
                       best_params=best_params)
        ans = "test_id,PM2.5,PM10,O3\n"
        ans_file = base_path_3 + "test/" + end_day + "-lightgbm_weather_log_params_" + best_params + "_" + type + "_" + str(
            feature_first) + ".csv"
        f_to = open(ans_file, 'wb')
        f_to.write(ans + ans1 + ans2)
        f_to.close()


def lightgbm_log_run(day1, day2, caiyun=False):
    # get_ans(type_="0301-0531_0801-0410", feature_first=False, start_day=day1, end_day=day2, best_params="1")
    # get_ans(type_="2017_0101-2018_0410_less", feature_first=False, start_day=day1, end_day=day2, best_params="1")
    # get_ans(type_="2017_0101-2018_0429_less", feature_first=False, start_day=day1, end_day=day2, best_params="1")
    # get_ans(type_="0301-0531_0801-0410", feature_first=False, start_day=day1, end_day=day2, best_params="2")
    # get_ans(type_="2017_0101-2018_0410_less", feature_first=False, start_day=day1, end_day=day2, best_params="2")
    # get_ans(type_="2017_0101-2018_0429_less", feature_first=False, start_day=day1, end_day=day2, best_params="2")
    # get_ans(type_="0301-0531_0801-0410", feature_first=False, start_day=day1, end_day=day2, best_params="3")
    # get_ans(type_="2017_0101-2018_0410_less", feature_first=False, start_day=day1, end_day=day2, best_params="3")
    # get_ans(type_="2017_0101-2018_0429_less", feature_first=False, start_day=day1, end_day=day2, best_params="3")
    # get_ans(type_="0301-0531_0801-0410", feature_first=False, start_day=day1, end_day=day2, best_params="4")
    # get_ans(type_="2017_0101-2018_0410_less", feature_first=False, start_day=day1, end_day=day2, best_params="4")
    # get_ans(type_="2017_0101-2018_0429_less", feature_first=False, start_day=day1, end_day=day2, best_params="4")

    get_ans(type_="2017_0101-2018_0515_less", feature_first=False, start_day=day1, end_day=day2, best_params="5", caiyun=caiyun)


if __name__ == '__main__':
    # city = "bj"
    # get_train_test_data(city=city)
    # city = "ld"
    # get_train_test_data(city=city)
    cities = ['bj', 'ld']
    attrs = ['PM25', 'PM10', 'O3']
    # params_file = open("lightgbm_best_params.txt", 'wb')
    # for city in cities:
    #     for attr in attrs:
    #         if city == "ld" and attr == 'O3':
    #             continue
    #         best_params = cal_best_params(city, attr, type="0301-0531_0801-0410", load_from_feature_file=True)
    #         params_file.write("city=" + city + ";attr=" + attr + "\nbest_params" + str(best_params) + "\n")
    # params_file.close()
    type = "2017_0101-2018_0515_less"
    # for city in cities:
    #     for attr in attrs:
    #         if city == "ld" and attr == 'O3':
    #             continue
    #         load_train_test(city, attr, type=type, load_from_feature_file=False)

    # for city in cities:
    #    for attr in attrs:
    #        if city == "ld" and attr == 'O3':
    #            continue
    #        score = train(city=city, attr=attr, best_params1="5", type=type,
    #                      load_from_feature_file=True)
    #        print score

    # type = "0301-0531_0801-0410"
    # for city in cities:
    #     for attr in attrs:
    #         if city == "ld" and attr == 'O3':
    #             continue
    #         score = train(city=city, attr=attr, best_params1="1", type=type,
    #                       load_from_feature_file=True)
    #         print score
    #
    # for city in cities:
    #     for attr in attrs:
    #         if city == "ld" and attr == 'O3':
    #             continue
    #         score = train(city=city, attr=attr, best_params1="2", type=type,
    #                       load_from_feature_file=True)
    #         print score
    # for city in cities:
    #     for attr in attrs:
    #         if city == "ld" and attr == 'O3':
    #             continue
    #         print score
    #         score = train(city=city, attr=attr, best_params1="2", type=type,
    #                       load_from_feature_file=True)
    #         print score

    # type = "0301-0531_0801-0410"
    # for city in cities:
    #     for attr in attrs:
    #         if city == "ld" and attr == 'O3':
    #             continue
    #         score = train(city=city, attr=attr, best_params1="1", type=type,
    #                       load_from_feature_file=True)
    #         print score
    # for city in cities:
    #     for attr in attrs:
    #         if city == "ld" and attr == 'O3':
    #             continue
    #         score = train(city=city, attr=attr, best_params1="2", type=type,
    #                       load_from_feature_file=True)
    #         print score


    time_now = datetime.now()
    time_now = time_now - timedelta(hours=12)
    start_day = (time_now - timedelta(days=2)).strftime('%Y-%m-%d')
    end_day = time_now.strftime('%Y-%m-%d')
    lightgbm_log_run(day1=start_day, day2=end_day, caiyun=True)
    lightgbm_log_run(day1=start_day, day2=end_day, caiyun=False)
    # xgboost_run(day1=start_day, day2=end_day)
    # get_ans(type_="0301-0531_0801-0410", feature_first=True, start_day=start_day, end_day=end_day)
    # get_ans(type_="0301-0531_0801-0410", feature_first=False, start_day=start_day, end_day=end_day)
    # get_ans(type_="2017_0101-2018_0410_less", feature_first=False, start_day=start_day, end_day=end_day)

    # get_test(type="0301-0531_0801-0410", feature_first=False, best_params="1")
    # get_test(type="2017_0101-2018_0410_less", feature_first=False, best_params="1")
    # get_test(type="2017_0101-2018_0429_less", feature_first=False, best_params="1")
    # get_test(type="0301-0531_0801-0410", feature_first=False, best_params="2")
    # get_test(type="2017_0101-2018_0410_less", feature_first=False, best_params="2")
    # get_test(type="2017_0101-2018_0429_less", feature_first=False, best_params="2")
