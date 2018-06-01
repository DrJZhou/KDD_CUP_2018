#!/usr/bin/python
# -*- coding: UTF-8 -*-
import bs4
import re
import requests
import io
import sys
import unicodedata
import time
from dateutil import parser
from dateutil.parser import parse
import json
import pandas as pd
from datetime import datetime, timedelta
# from data_processing import *
from draw import *
import xlrd
from unit import *

reload(sys)
sys.setdefaultencoding('utf-8')
base_path_1 = "../dataset/"
base_path_2 = "../dataset/tmp/"
base_path_3 = "../output/"

'''
KC1 用于 填补 KF1的缺失值，KC1不需要预测
zhiwuyuan_aq 用 wanliu的预测结果，同时wanliu的数据用于填补zhiwuyuan
'CD1' ！= 'MY7'
'''

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


# def analysis_station():
#     stations = load_station()
#     nearst = {}
#
#     for city in stations.keys():
#         city_stations = stations[city]
#         stations_list = []
#         stations_mask = np.zeros(len(city_stations.keys()))
#         for station_id in city_stations.keys():
#             if station_id == "KC1":
#                 continue
#             stations_list.append([station_id, city_stations[station_id]["lat"], city_stations[station_id]["lng"]])
#         for i in range(len(stations_list)):
#             # if stations_mask[i] == 1:
#             #     continue
#             min_distance = 100000000
#             flag = 0
#             for j in range(0, len(stations_list)):
#                 # print stations_list[i], stations_list[j]
#                 if i == j:
#                     continue
#                 distance = Distance1(stations_list[i][1], stations_list[i][2], stations_list[j][1], stations_list[j][2])
#                 if distance < min_distance:
#                     min_distance = distance
#                     flag = j
#             # stations_mask[i] = stations_mask[flag] = 1
#             nearst[stations_list[i][0]] = stations_list[flag][0]
#             # nearst[stations_list[i][0]] = stations_list[flag][0]
#     # print nearst
#     for key in nearst.keys():
#         print key, nearst[key]


def get_holiday_weekend(day_today="2018-04-30"):
    # print "day:", day
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

    year = int(day_today.split("-")[0])
    month = int(day_today.split("-")[1])
    day = int(day_today.split("-")[2])

    tmp = []
    day_now = datetime(year, month, day)
    day = day_now
    day_str = datetime_toString(day)
    # 判断是否周末
    week = day.weekday()
    # print day
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
    return tmp


# def main():
#     city = 'bj'
#     time_now = time.time()
#     time_now = time.localtime(time_now)
#     time_now = time.strftime('%Y-%m-%d', time_now)
#     # stations = load_station()
#     # print stations
#     # start_time = str(time_now)+"-0"
#     start_time = "2018-04-01-0"
#     end_time = "2018-04-10-23"
#     # end_time = str(time_now) + "-23"
#     # get_data(city, start_time, end_time)
#     # df = load_data(city, stations, start_time, end_time)
#     # analysis(df, stations, city)
#     # city = 'bj'
#     # df = load_data(city, start_time, end_time)
#     # analysis(df, stations, city)

def rule_weight(filename1, filename2, a, b, c):
    df1 = pd.read_csv(filename1, sep=',')
    df1.index = df1['test_id']
    df1.iloc[:, 1] = df1.iloc[:, 1] * a
    df1.iloc[:, 2] = df1.iloc[:, 2] * b
    df1.iloc[:, 3] = df1.iloc[:, 3] * c
    df1 = df1.drop(["test_id"], axis=1)
    df1.to_csv(filename2, index=True, sep=',')


def rule_int(filename1, filename2):
    df1 = pd.read_csv(filename1, sep=',')
    df1.index = df1['test_id']
    for i in range(df1.values.shape[0]):
        df1.iloc[i, 1] = int(df1.values[i, 1] + 0.5)
        df1.iloc[i, 2] = int(df1.values[i, 2] + 0.5)
        df1.iloc[i, 3] = int(df1.values[i, 3] + 0.5)
    df1 = df1.drop(["test_id"], axis=1)
    df1.to_csv(filename2, index=True, sep=',')


def rule_run(end_day):
    oneday_after_end_day = datetime_toString(string_toDatetime(end_day) + timedelta(days=1))
    twoday_after_end_day = datetime_toString(string_toDatetime(end_day) + timedelta(days=2))
    tmp1 = get_holiday_weekend(oneday_after_end_day)
    tmp2 = get_holiday_weekend(twoday_after_end_day)
    print tmp1, tmp2
    # 0 判断是否周末
    # 1 判断是否节假日
    # 2 判断是否工作
    # 3 判断放假最后一天
    # 4 判断上班第一天
    # 5 判断上班最后一天
    # 6 判断放假第一天
    filename = base_path_3 + "ensemble_all.csv"
    df1 = pd.read_csv(filename, sep=',')
    df1.index = df1['test_id']

    # #
    # print df1.size
    # print df1.iloc[0, 0]
    # print df1
    for i in range(df1.values.shape[0]):
        # print df1.iloc[i, 0]
        station_id = df1.iloc[i, 0].split("#")[0]
        if len(station_id) < 5:
            continue
        hour = int(df1.iloc[i, 0].split("#")[1])
        if tmp1[5] == 1:
            # 7-11
            if hour >= 7 and hour <= 11:
                df1.iloc[i, 1:] = df1.iloc[i, 1:] * 1.05
        if tmp2[5] == 1:
            if hour >= 7 + 24 and hour <= 11 + 24:
                df1.iloc[i, 1:] = df1.iloc[i, 1:] * 1.05
        # if tmp1[4] == 1:
        #     if hour >= 0 and hour <= 10:
        #         df1.iloc[i, 1:] = df1.iloc[i, 1:] * 1.10
        # if tmp2[4] == 1:
        #     if hour >= 0+24 and hour <= 10+24:
        #         df1.iloc[i, 1:] = df1.iloc[i, 1:] * 1.10
    df1 = df1.drop(["test_id"], axis=1)
    filename1 = base_path_3 + "ensemble_all_rule.csv"
    df1.to_csv(filename1, index=True, sep=',')


if __name__ == '__main__':
    # analysis_station()
    # main()
    rule_run(end_day='2018-04-28')
    # print get_holiday_weekend(day_today='2018-05-01')
