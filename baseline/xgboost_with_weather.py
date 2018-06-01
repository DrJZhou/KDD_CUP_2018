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
from weather_data_processing import *

reload(sys)
sys.setdefaultencoding('utf-8')
base_path_1 = "../dataset/"
base_path_2 = "../dataset/tmp/"
base_path_3 = "../output/"

station_id_change = {
    'miyunshuiku_aq': 'miyunshuik_aq',
    'wanshouxigong_aq': 'wanshouxig_aq',
    'nongzhanguan_aq': 'nongzhangu_aq',
    'xizhimenbei_aq': 'xizhimenbe_aq',
    'fengtaihuayuan_aq': 'fengtaihua_aq',
    'aotizhongxin_aq': 'aotizhongx_aq',
    'yongdingmennei_aq': 'yongdingme_aq'
}


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
        np.savetxt(base_path_2 + city + "_" + attr + '_training_feature_X_weather_' + type + '_1.csv', X, delimiter=',')
    else:
        X = np.loadtxt(base_path_2 + city + "_" + attr + '_training_feature_X_weather_' + type + '_1.csv',
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
    ans_X, ans_Y = change_X_Y(np.concatenate([X[:7900, :], X[-7900:, :]]), np.concatenate([Y[:7900, :], Y[-7900:, :]]))
    # ans_X, ans_Y = change_X_Y(np.concatenate([X[:, :], X[:, :]]), np.concatenate([Y[:, :], Y[:, :]]))
    train_X, test_X, train_Y, test_Y = train_test_split(ans_X, ans_Y, test_size=0.02, random_state=11)
    print train_X.shape, test_X.shape, train_Y.shape, test_Y.shape
    return train_X, test_X, train_Y, test_Y


def get_score(pred, valid_y_exp):
    return np.mean(np.abs(pred - valid_y_exp) / (pred + valid_y_exp) * 2)


params = {
    'max_depth': range(7, 12, 2),
    'min_child_weight': [2],
    'gamma': [i / 10.0 for i in range(8, 9)],
    'subsample': [i / 10.0 for i in range(6, 10)],
    'colsample_bytree': [i / 10.0 for i in range(6, 10)],
    'reg_alpha': [0, 0.001, 0.001],
    'learning_rate': [0.001, 0.01, 0.1],
    'n_estimators': [1000, 2000, 3000]
}
cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=2)
CLF = GridSearchCV(
    estimator=xgb.XGBRegressor(learning_rate=0.001, n_estimators=3000, max_depth=10, min_child_weight=2,
                               reg_alpha=0.001, gamma=0.6, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1,
                               seed=27), param_grid=params, scoring=scoring, n_jobs=-1, cv=cv, verbose=6)


def params(city='bj', attr="PM25"):
    train_X, test_X, train_Y, test_Y = load_train_test(city=city, attr=attr)
    # RF.fit(train_X, train_Y)
    CLF.fit(train_X, train_Y)
    print 'best params:\n', CLF.best_params_
    mean_scores = np.array(CLF.cv_results_['mean_test_score'])
    print 'mean score', mean_scores
    print 'best score', CLF.best_score_
    return CLF.best_params_


def train(city, attr, best_params1=None, type="0301-0531_0801-0410", load_from_feature_file=False):
    if best_params1 is None:
        best_params1 = {
            'max_depth': 10,
            'learning_rate': 0.001,
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
    train_X, test_X, train_Y, test_Y = load_train_test(city=city, attr=attr, type=type,
                                                       load_from_feature_file=load_from_feature_file)
    reg = xgb.XGBRegressor(**best_params1)
    reg.fit(train_X, train_Y, eval_set=[(train_X, train_Y), (test_X, test_Y)], verbose=100,
            early_stopping_rounds=20)
    test_Y1 = reg.predict(test_X)
    score = get_score(test_Y1, test_Y)
    model_file = base_path_2 + city + '_' + attr + '_best_xgboost_with_weather_' + type + '_1.model'
    joblib.dump(reg, model_file)
    return score


def change_feature(feature):
    ans = []
    for i in range(48):
        tmp = np.zeros(48)
        tmp[i] = 1
        ans.append(np.hstack((feature, np.array(tmp))))
    return np.array(ans)


def predict(city, length=24 * (3 * 7 + 2), start_day="2018-04-11", end_day="2018-04-11", type="0301-0531_0801-0410",
            feature_first=False, caiyun=False):
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
        model_PM25_file = base_path_2 + city + '_PM25_best_xgboost_with_weather_' + type + '_1.model'
        model_PM10_file = base_path_2 + city + '_PM10_best_xgboost_with_weather_' + type + '_1.model'
    else:
        model_PM25_file = base_path_2 + city + '_PM25_best_xgboost_with_weather_' + type + '.model'
        model_PM10_file = base_path_2 + city + '_PM10_best_xgboost_with_weather_' + type + '.model'
    model_PM25 = joblib.load(model_PM25_file)
    model_PM10 = joblib.load(model_PM10_file)
    if city == "bj":
        if feature_first == False:
            model_O3_file = base_path_2 + city + '_O3_best_xgboost_with_weather_' + type + '_1.model'
        else:
            model_O3_file = base_path_2 + city + '_O3_best_xgboost_with_weather_' + type + '.model'
        model_O3 = joblib.load(model_O3_file)
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
        congfu = weather_data_tmp.groupby('time').apply(
            lambda d: tuple(d.index) if len(d.index) > 1 else None
        ).dropna()
        print congfu
        weather_data_tmp = weather_data_tmp.drop_duplicates(['time'])
        values = group[attr_need].values
        weather_values = weather_data_tmp[weather_attr_need].values
        # print values.shape, weather_values.shape
        # print weather_data_tmp[weather_attr_need]
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
        pred_PM25 = model_PM25.predict(change_feature(PM25_feature[0]))
        pred_PM10 = model_PM10.predict(change_feature(PM10_feature[0]))
        if station_id_change.has_key(station):
            station = station_id_change[station]
        if city == "bj":
            if feature_first == False:
                O3_feature = get_all_feature(np.array([tmp]), city, attr="O3")
            else:
                O3_feature = get_all_feature_1(np.array([tmp]), city, attr="O3")
            pred_O3 = model_O3.predict(change_feature(O3_feature[0]))
            for i in range(48):
                ans += station + "#" + str(i) + "," + str(pred_PM25[i]) + "," + str(pred_PM10[i]) + "," + str(
                    pred_O3[i]) + "\n"
                # print tmp.shape
        else:
            for i in range(48):
                ans += station + "#" + str(i) + "," + str(pred_PM25[i]) + "," + str(pred_PM10[i]) + ",0.0\n"
    return ans


def get_ans(type_, feature_first=False, start_day="2018-04-27", end_day="2018-04-29", caiyun=False):
    # ans1 = ""
    # start_day = "2018-04-27"
    # end_day = "2018-04-29"
    ans1 = predict(city='bj', start_day=start_day, end_day=end_day, type=type_, feature_first=feature_first,
                   caiyun=caiyun)
    ans2 = predict(city='ld', start_day=start_day, end_day=end_day, type=type_, feature_first=feature_first,
                   caiyun=caiyun)
    ans = "test_id,PM2.5,PM10,O3\n"
    if caiyun == False:
        ans_file = base_path_3 + end_day + "-xgboost_weather" + type_ + "_" + str(feature_first) + ".csv"
    else:
        ans_file = base_path_3 + end_day + "-xgboost_weather" + type_ + "_" + str(feature_first) + "_caiyun.csv"
    f_to = open(ans_file, 'wb')
    f_to.write(ans + ans1 + ans2)
    f_to.close()


def get_test(type, feature_first=False, caiyun=False):
    print type
    for i in range(11, 28):
        start_day = "2018-04-" + str(i)
        end_day = "2018-04-" + str(i)
        ans1 = predict(city='bj', start_day=start_day, end_day=end_day, type=type, feature_first=feature_first, caiyun=caiyun)
        ans2 = predict(city='ld', start_day=start_day, end_day=end_day, type=type, feature_first=feature_first, caiyun=caiyun)
        ans = "test_id,PM2.5,PM10,O3\n"
        ans_file = base_path_3 + "test/" + end_day + "-xgboost_weather" + type + "_" + str(feature_first) + ".csv"
        f_to = open(ans_file, 'wb')
        f_to.write(ans + ans1 + ans2)
        f_to.close()


def xgboost_run(day1, day2, caiyun=False):
    get_ans(type_="0301-0531_0801-0410", feature_first=True, start_day=day1, end_day=day2, caiyun=caiyun)
    get_ans(type_="0301-0531_0801-0410", feature_first=False, start_day=day1, end_day=day2, caiyun=caiyun)
    get_ans(type_="2017_0101-2018_0410_less", feature_first=False, start_day=day1, end_day=day2, caiyun=caiyun)


if __name__ == '__main__':
    # city = "bj"
    # get_train_test_data(city=city)
    # city = "ld"
    # get_train_test_data(city=city)
    cities = ['bj', 'ld']
    attrs = ['PM25', 'PM10', 'O3']
    # params_file = open("xgboost_best_params.txt", 'wb')
    # for city in cities:
    #     for attr in attrs:
    #         if city == "ld" and attr == 'O3':
    #             continue
    #         best_params = params(city=city, attr=attr)
    #         params_file.write("city=" + city + ";attr=" + attr + "\nbest_params" + str(best_params) + "\n")
    # params_file.close()

    type = "0301-0531_0801-0410"
    type = "0201-0531_0801-0429"
    '''
    0.447176528537
    0.56958264329
    0.622310882246
    0.537411449426
    0.414000223121
    0.402384721824
    0.423329844649
    0.401235642362
    0.474126078298
    0.871000288918
    0.656048580397
    0.566214029215
    0.52515431741
    '''

    # type = "0101-0410"
    # type = "2017_0101-2018_0410_less"
    '''
    0.436929029956
    0.539369699451
    0.54247770043
    0.540824182611
    0.43677272213
    0.411579081144
    0.427990481448
    0.351758546148
    0.457626718596
    0.827787082369
    0.74869960649
    0.575685943214
    '''
    # for city in cities:
    #     for attr in attrs:
    #         if city == "bj":
    #             continue
    #         if city == "ld" and attr == 'O3':
    #             continue
    #         if city == "ld" and attr == 'PM25':
    #             continue
    #         score = train(city=city, attr=attr, best_params1=None, type=type,
    #                       load_from_feature_file=True)  # , best_params1=best_params[city][attr]
    #         print score
    # time_now = datetime.now()
    # time_now = time_now - timedelta(hours=8)
    # start_day = (time_now - timedelta(days=2)).strftime('%Y-%m-%d')
    # end_day = time_now.strftime('%Y-%m-%d')
    # xgboost_run(day1=start_day, day2=end_day)
    # get_ans(type_="0301-0531_0801-0410", feature_first=True, start_day=start_day, end_day=end_day)
    # get_ans(type_="0301-0531_0801-0410", feature_first=False, start_day=start_day, end_day=end_day)
    # get_ans(type_="2017_0101-2018_0410_less", feature_first=False, start_day=start_day, end_day=end_day)
    #
    # import model
    # type = "2017_0101-2018_0410_less"
    # print type
    # for i in range(11, 24):
    #     ans1 = ""
    #     start_day = "2018-04-" + str(i)
    #     end_day = "2018-04-" + str(i)
    #     ans1 = predict(city='bj', start_day=start_day, end_day=end_day, type=type)
    #     ans2 = predict(city='ld', start_day=start_day, end_day=end_day, type=type)
    #     ans = "test_id,PM2.5,PM10,O3\n"
    #     ans_file = base_path_3 + end_day + "-V1.csv"
    #     f_to = open(ans_file, 'wb')
    #     f_to.write(ans + ans1 + ans2)
    #     f_to.close()
    #
    # for i in range(11, 23):
    #         start_day = "2018-04-" + str(i)
    #         model.test(start_day)
    # get_test(type="0301-0531_0801-0410", feature_first=True)
    # get_test(type="0301-0531_0801-0410", feature_first=False)
    # get_test(type="2017_0101-2018_0410_less", feature_first=False)
