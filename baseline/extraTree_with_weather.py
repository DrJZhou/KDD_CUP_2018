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
from weather_data_processing import *
from unit import *
import cPickle as pickle

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
        attr_need = ["PM25_Concentration", "PM10_Concentration", "O3_Concentration"] \
                    + ['time_year', 'time_month', 'time_day', 'time_week', 'time_hour']
    else:
        attr_need = ["PM25_Concentration", "PM10_Concentration"] \
                    + ['time_year', 'time_month', 'time_day', 'time_week', 'time_hour']

    stations = load_station()
    filename = base_path_2 + "rate.pkl"
    f1 = file(filename, 'rb')
    loss_rate = pickle.load(f1)

    ans_history = history_data(city=city, stations=stations, start_day="2017-01-01", end_day="2018-04-10")
    weather_history = history_weather_data(city=city, start_day="2017-01-01", end_day="2018-04-10")
    ans_current = get_all_processing_data(city, start_day="2018-04-11", end_day="2018-05-15")
    weather_current = load_all_weather_data(city, start_day="2018-04-11", end_day="2018-05-15", crawl_data=False)
    weather_data = pd.concat([weather_history, weather_current["2018-04-11":"2018-05-15"]], axis=0)
    weather_groups = weather_data.groupby("station_id")
    # print ans_history
    ans = []
    for station, group in ans_history.items():
        grid_station = nearst[city][station]
        weather_data = weather_groups.get_group(grid_station).sort_index()
        station_num_id = stations[city][station]["station_num_id"]
        station_type_id = stations[city][station]["type_id"]
        if ans_current.has_key(station):
            group = pd.concat([ans_history[station], ans_current[station]["2018-04-11":"2018-05-15"]],
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
                rate_flag = 0
                avg_rate1 = 0.0
                avg_rate2 = 0.0
                avg_rate3 = 0.0
                for j in range(23):
                    year1 = values[i + length - 24 * (j + 1), 3]
                    month1 = values[i + length - 24 * (j + 1), 4]
                    day1 = values[i + length - 24 * (j + 1), 5]
                    day = "%d-%02d-%02d" % (int(year1), int(month1), int(day1))
                    if j < 2:
                        if loss_rate[city][station][day]['all'] > 0.5:
                            rate_flag = 1
                            break
                    elif j < 9:
                        avg_rate1 += loss_rate[city][station][day]['all']
                    elif j < 16:
                        avg_rate2 += loss_rate[city][station][day]['all']
                    else:
                        avg_rate3 += loss_rate[city][station][day]['all']
                if avg_rate1 / 7 > 0.60:
                    continue
                if avg_rate2 / 7 > 0.7:
                    continue
                if avg_rate3 / 7 > 0.75:
                    continue
                if rate_flag == 1:
                    continue
                pass
            else:
                rate_flag = 0
                avg_rate1 = 0.0
                avg_rate2 = 0.0
                avg_rate3 = 0.0
                for j in range(23):
                    year1 = values[i + length - 24 * (j + 1), 2]
                    month1 = values[i + length - 24 * (j + 1), 3]
                    day1 = values[i + length - 24 * (j + 1), 4]
                    day = "%d-%02d-%02d" % (int(year1), int(month1), int(day1))
                    if j < 2:
                        if loss_rate[city][station][day]['all'] > 0.50:
                            rate_flag = 1
                            break
                    elif j < 9:
                        avg_rate1 += loss_rate[city][station][day]['all']
                    elif j < 16:
                        avg_rate2 += loss_rate[city][station][day]['all']
                    else:
                        avg_rate3 += loss_rate[city][station][day]['all']
                if avg_rate1 / 7 > 0.60:
                    continue
                if avg_rate2 / 7 > 0.70:
                    continue
                if avg_rate3 / 7 > 0.75:
                    continue
                if rate_flag == 1:
                    continue
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
                continue
            ans.append(tmp)
    ans = np.array(ans)
    # 170810-0410
    np.savetxt(base_path_2 + city + '_training_weather_2017_0101-2018_0515_less.csv', ans, delimiter=',')
    # 170810-0410  03010531
    # np.savetxt(base_path_2 + city + '_training_weather_three_metric0801_03010531.csv', ans, delimiter=',')


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
    # ans = np.hstack((mean_, median_, max_, min_, var_, std_))
    ans = ans.reshape(-1, 7)
    # ans = ans.reshape(-1, 6)
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
        # ans.append(np.hstack((mean_, median_, max_, min_, var_, std_)))
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
        # for i in range(7, int(data.shape[1] / 24)):
        day_static = np.hstack((day_static, get_statistic_feature(data[:, i * 24:(i + 1) * 24])))
    week_static = np.array([[] for i in range(data.shape[0])])
    for i in range(int(data.shape[1] / (7 * 24))):
        # week_static = np.hstack((week_static, get_statistic_feature(data[:, i * 24:(i + 1) * 24])))
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
        # day_now = datetime(year, month, day) - timedelta(days=6)
        for j in range(23):
            # for j in range(7):
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
    if city == 'bj':
        holiday_weekend_feature = get_holiday_weekend(data[:, 2:6])
    if attr == "PM25":
        static_feature = get_all_statistic_feature_1(data[:, 6: 6 + length])
    elif attr == "PM10":
        static_feature = get_all_statistic_feature_1(data[:, 6 + length: 6 + length * 2])
    else:
        static_feature = get_all_statistic_feature_1(data[:, 6 + length * 2: 6 + length * 3])
    if city == "bj":
        all_feature = np.hstack(
            (data[:, 6:], holiday_weekend_feature, onehot_feature, static_feature))
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
    #     [data[:, 6 + length * 3 + 18 * 24: 6 + length * 3 + (length + 24 * 2)],
    #      data[:, 6 + length * 3 + (length + 24 * 2) + 18 * 24: 6 + length * 3 + (length + 24 * 2) * 2],
    #      data[:, 6 + length * 3 + (length + 24 * 2) * 2 + 18 * 24: 6 + length * 3 + (length + 24 * 2) * 3]])
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


def load_train_test(city, attr, type="0301-0531_0801-0410", load_from_feature_file=False):
    filename = base_path_2 + city + '_training_weather_' + type + '.csv'
    data = np.loadtxt(filename, delimiter=",")
    # data = data[:1000, ]
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
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.001, random_state=11)
    print train_X.shape, test_X.shape, train_Y.shape, test_Y.shape
    return train_X, test_X, train_Y, test_Y


def get_score(pred, valid_y_exp):
    return np.mean(np.abs(pred - valid_y_exp) / (pred + valid_y_exp) * 2)


params = {'n_estimators': [100, 500, 1000], 'max_depth': [5, 10, 15, 20],
          'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2, 5],
          'max_features': ['auto', 'sqrt', 'log2', None]}
# LR = LinearRegression()
# SVR = LinearSVR()
RF = RandomForestRegressor()
EXT = ExtraTreesRegressor(n_jobs=-1)
cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=2)
CLF = GridSearchCV(EXT, params, scoring=scoring, verbose=6, cv=cv)


def params(city='bj', attr="PM25", type="0301-0531_0801-0410", load_from_feature_file=False):
    train_X, test_X, train_Y, test_Y = load_train_test(city=city, attr=attr, type=type,
                                                       load_from_feature_file=load_from_feature_file)
    # RF.fit(train_X, train_Y)
    CLF.fit(train_X[:1000, ], train_Y[:1000, ])
    print 'best params:\n', CLF.best_params_
    mean_scores = np.array(CLF.cv_results_['mean_test_score'])
    print 'mean score', mean_scores
    print 'best score', CLF.best_score_
    return CLF.best_params_


def train(city, attr, best_params1=None, type="0301-0531_0801-0410", load_from_feature_file=False):
    if best_params1 is None:
        best_params1 = {'max_features': None, 'min_samples_split': 2, 'n_estimators': 100, 'max_depth': 15,
                        'min_samples_leaf': 1}
    train_X, test_X, train_Y, test_Y = load_train_test(city=city, attr=attr, type=type,
                                                       load_from_feature_file=load_from_feature_file)
    EXT1 = ExtraTreesRegressor(n_jobs=-1, random_state=1, **best_params1)
    # train_X = np.concatenate([train_X, test_X])
    # train_Y = np.concatenate([train_Y, test_Y])
    # train_X = np.nan_to_num(train_X)
    # train_Y = np.nan_to_num(train_Y)
    EXT1.fit(train_X, train_Y)
    test_Y1 = EXT1.predict(test_X)
    score = get_score(test_Y1, test_Y)
    model_file = base_path_2 + city + '_' + attr + '_best_extraTree_with_weather_' + type + '_1.model'
    joblib.dump(EXT1, model_file)
    return score


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
    # print ans_current
    weather_history = history_weather_data(city=city, start_day="2018-01-01", end_day="2018-04-10")
    weather_current = load_all_weather_data(city, start_day=start_day, end_day=end_day, crawl_data=False, caiyun=caiyun)
    # print weather_current
    weather_data = pd.concat([weather_history, weather_current], axis=0)
    weather_data = weather_data.drop_duplicates(["station_id", 'time'])
    weather_groups = weather_data.groupby("station_id")
    if feature_first == False:
        model_PM25_file = base_path_2 + city + '_PM25_best_extraTree_with_weather_' + type + '_1.model'
    else:
        model_PM25_file = base_path_2 + city + '_PM25_best_extraTree_with_weather_' + type + '.model'
    model_PM25 = joblib.load(model_PM25_file)
    if feature_first == False:
        model_PM10_file = base_path_2 + city + '_PM10_best_extraTree_with_weather_' + type + '_1.model'
    else:
        model_PM10_file = base_path_2 + city + '_PM10_best_extraTree_with_weather_' + type + '.model'
    model_PM10 = joblib.load(model_PM10_file)
    if city == "bj":
        if feature_first == False:
            model_O3_file = base_path_2 + city + '_O3_best_extraTree_with_weather_' + type + '_1.model'
        else:
            model_O3_file = base_path_2 + city + '_O3_best_extraTree_with_weather_' + type + '.model'
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
        # print ans_current[station]["2018-04-11":]
        group = pd.concat([ans_history[station], ans_current[station]["2018-04-11":]], axis=0).sort_index()
        # print group
        grid_station = nearst[city][station]
        weather_data_tmp = weather_groups.get_group(grid_station).sort_index()
        weather_data_tmp = weather_data_tmp.drop_duplicates(['time'])
        # print weather_data_tmp
        # print group
        values = group[attr_need].values
        weather_values = weather_data_tmp[weather_attr_need].values
        # print values.shape, weather_values.shape
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
        pred_PM25 = model_PM25.predict(PM25_feature)[0]
        pred_PM10 = model_PM10.predict(PM10_feature)[0]
        if station_id_change.has_key(station):
            station = station_id_change[station]
        if city == "bj":
            if feature_first == False:
                O3_feature = get_all_feature(np.array([tmp]), city, attr="O3")
            else:
                O3_feature = get_all_feature_1(np.array([tmp]), city, attr="O3")
            pred_O3 = model_O3.predict(O3_feature)[0]
            for i in range(48):
                ans += station + "#" + str(i) + "," + str(pred_PM25[i]) + "," + str(pred_PM10[i]) + "," + str(
                    pred_O3[i]) + "\n"
                # print tmp.shape
        else:
            for i in range(48):
                ans += station + "#" + str(i) + "," + str(pred_PM25[i]) + "," + str(pred_PM10[i]) + ",0.0\n"
    return ans


def get_ans(type, feature_first=False, start_day="2018-04-27", end_day="2018-04-29", caiyun=False):
    ans1 = predict(city='bj', start_day=start_day, end_day=end_day, type=type, feature_first=feature_first,
                   caiyun=caiyun)
    ans2 = predict(city='ld', start_day=start_day, end_day=end_day, type=type, feature_first=feature_first,
                   caiyun=caiyun)
    ans = "test_id,PM2.5,PM10,O3\n"
    if caiyun == False:
        ans_file = base_path_3 + end_day + "-ext_with_weather_" + type + "_" + str(feature_first) + ".csv"
    else:
        ans_file = base_path_3 + end_day + "-ext_with_weather_" + type + "_" + str(feature_first) + "_caiyun.csv"
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
        ans_file = base_path_3 + "test/" + end_day + "-ext_with_weather_" + type + "_" + str(feature_first) + ".csv"
        f_to = open(ans_file, 'wb')
        f_to.write(ans + ans1 + ans2)
        f_to.close()


def ext_with_weather_run(day1, day2, caiyun=False):
    # get_ans(type="0301-0531_0801-0410", feature_first=False, start_day=day1, end_day=day2)
    get_ans(type="0301-0531_0801-0410", feature_first=False, start_day=day1, end_day=day2, caiyun=caiyun)
    # get_ans(type="0301-0531_0801-0410", feature_first=True, start_day=day1, end_day=day2)
    # get_ans(type="2017_0101-2018_0410", feature_first=True, start_day=day1, end_day=day2)
    get_ans(type="2017_0101-2018_0410_less", feature_first=False, start_day=day1, end_day=day2, caiyun=caiyun)


if __name__ == '__main__':
    # city = "bj"
    # get_train_test_data(city=city)
    # city = "ld"
    # get_train_test_data(city=city)
    cities = ['bj', 'ld']
    attrs = ['PM25', 'PM10', 'O3']
    type = "2017_0101-2018_0410_less"
    # params_file = open("best_params"+ type +".txt", 'wb')
    # for city in cities:
    #     for attr in attrs:
    #         if city == "ld" and attr == 'O3':
    #             continue
    #         best_params = params(city=city, attr=attr, type=type, load_from_feature_file=True)
    #         params_file.write("city=" + city + ";attr=" + attr + "\nbest_params" + str(best_params) + "\n")
    # params_file.close()

    best_params = {
        "bj": {
            "PM25": {'max_features': None, 'min_samples_split': 2, 'n_estimators': 500, 'max_depth': 20,
                     'min_samples_leaf': 1},
            "PM10": {'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 500, 'max_depth': 20,
                     'min_samples_leaf': 1},
            "O3": {'max_features': None, 'min_samples_split': 2, 'n_estimators': 500, 'max_depth': 20,
                   'min_samples_leaf': 1}
        },
        "ld": {
            "PM25": {'max_features': 'auto', 'min_samples_split': 2, 'n_estimators': 500, 'max_depth': 20,
                     'min_samples_leaf': 1},
            "PM10": {'max_features': None, 'min_samples_split': 2, 'n_estimators': 500, 'max_depth': 20,
                     'min_samples_leaf': 1}
        }
    }
    # type = "0301-0531_0801-0410"
    '''
    11 0.445231744031
    12 0.586129349615
    13 0.644352768339
    14 0.488263282678
    15 0.375989864638
    16 0.381937764082
    17 0.406635665847
    18 0.339953732065
    19 0.456904011911
    20 0.865736767424
    21 0.76071451531
    22 0.658287533912
    
    
    ---------------------
    0.422964438994
    0.540424375083
    0.57841564512
    0.480312106
    0.448210426774
    0.384829206374
    0.376893368389
    0.319712178625
    0.437614578311
    0.859329760293
    0.730307511289
    0.649798314065
    '''

    # type = "2017_0101-2018_0410"
    '''
    0.451319821172 0.411533329795 0.525747007714
    0.593397473117 0.465503700196 0.669104126386
    0.617187167171 0.597554120984 0.637112796011
    0.474734913222 0.55534097057 0.383674373426
    0.402423567518 0.394966098754 0.410281541475
    0.372899236556 0.295070782578 0.451570597977
    0.404621528849 0.413350015831 0.394385429714
    0.333472898578 0.325422459383 0.341763966286
    0.537935360988 0.433002156902 0.861889302248
    0.895274493627 0.804449507801 0.936030578777
    0.778402429703 0.720367195117 0.822438862249
    0.664834671091 0.605123504737 0.738706814966

    '''
    type = "2017_0101-2018_0410_less"
    '''
    0.438083523776
    0.543805846611
    0.559071764979
    0.469470814224
    0.452514480028
    0.406012494206
    0.38386749315
    0.325740281555
    0.434321778386
    0.857568580213
    0.733842150071
    0.697184890254
    '''

    # for city in cities:
    #     for attr in attrs:
    #         if city == "ld" and attr == 'O3':
    #             continue
    #         score = train(city=city, attr=attr, best_params1=best_params[city][attr], type=type,
    #                       load_from_feature_file=True)
    #         print score

    # type = "0301-0531_0801-0410"
    type = "2017_0101-2018_0429_less"
    for city in cities:
        for attr in attrs:
            if city == "ld" and attr == 'O3':
                continue
            load_train_test(city, attr, type=type, load_from_feature_file=False)

    for city in cities:
        for attr in attrs:
            if city == "ld" and attr == 'O3':
                continue
            score = train(city=city, attr=attr, best_params1=best_params[city][attr], type=type,
                          load_from_feature_file=True)
            print score
    # type = "2017_0101-2018_0410_less"

    # for city in cities:
    #     for attr in attrs:
    #         if city == "ld" and attr == 'O3':
    #             continue
    #         load_train_test(city, attr, type=type, load_from_feature_file=False)
    # for city in cities:
    #     for attr in attrs:
    #         if city == "ld" and attr == 'O3':
    #             continue
    #         score = train(city=city, attr=attr, best_params1=best_params[city][attr], type=type,
    #                       load_from_feature_file=True)  # True
    #         print score
    #
    # time_now = datetime.now()
    # time_now = time_now - timedelta(hours=8)
    # start_day = (time_now - timedelta(days=2)).strftime('%Y-%m-%d')
    # end_day = time_now.strftime('%Y-%m-%d')
    # get_ans(type="0301-0531_0801-0410", feature_first=False, start_day=start_day, end_day=end_day)
    # get_ans(type="0301-0531_0801-0410", feature_first=True, start_day=start_day, end_day=end_day)
    # get_ans(type="2017_0101-2018_0410", feature_first=True, start_day=start_day, end_day=end_day)
    # get_ans(type="2017_0101-2018_0410_less", feature_first=False, start_day=start_day, end_day=end_day)
    #
    # import model
    #
    # print type
    # for i in range(11, 23):
    #     ans1 = ""
    #     start_day = "2018-04-" + str(i)
    #     end_day = "2018-04-" + str(i)
    #     ans1 = predict(city='bj', start_day=start_day, end_day=end_day, type=type)
    #     ans2 = predict(city='ld', start_day=start_day, end_day=end_day, type=type)
    #     ans = "test_id,PM2.5,PM10,O3\n"
    #     ans_file = base_path_3 + end_day + "-V4.csv"
    #     f_to = open(ans_file, 'wb')
    #     f_to.write(ans + ans1 + ans2)
    #     f_to.close()
    #
    # for i in range(11, 23):
    #     start_day = "2018-04-" + str(i)
    #     model.test(start_day)

    # get_test(type="0301-0531_0801-0410", feature_first=False)
    # get_test(type="0301-0531_0801-0410", feature_first=True)
    # get_test(type="2017_0101-2018_0410", feature_first=True)
    # get_test(type="2017_0101-2018_0410_less", feature_first=False)
    get_test(type="2017_0101-2018_0429_less", feature_first=False)
