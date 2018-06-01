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
from weather_data_processing import *
# from draw import *
import xlrd

reload(sys)
sys.setdefaultencoding('utf-8')
base_path_1 = "../dataset/"
base_path_2 = "../dataset/tmp/"
base_path_3 = "../output/"


# base_path_2 = "./"


# 从网站下载数据
def get_data(city, start_time, end_time, current_day=False):
    if current_day == True:
        end_time = '2018-07-01-23'
    link1 = 'https://biendata.com/competition/airquality/' + city + '/' + start_time + '/' + end_time + '/2k0d1d8'
    respones = requests.get(link1)
    if current_day == False:
        with open(base_path_2 + city + "_airquality_" + start_time + "_" + end_time + ".csv", 'w') as f:
            f.write(respones.text)
    else:
        with open(base_path_2 + city + "_airquality_current_day.csv", 'w') as f:
            f.write(respones.text)
    print "get " + city + " air quality data"


# 从网站下载数据
def get_weather_data(city, start_time, end_time, current_day=False):
    if current_day == True:
        end_time = "2018-07-01-23"
    link3 = 'https://biendata.com/competition/meteorology/' + city + '_grid/' + start_time + '/' + end_time + '/2k0d1d8'
    respones = requests.get(link3)
    if current_day == False:
        with open(base_path_2 + city + "_meteorology_grid_" + start_time + "_" + end_time + ".csv", 'w') as f:
            f.write(respones.text)
    else:
        with open(base_path_2 + city + "_meteorology_grid_current_day.csv", 'w') as f:
            f.write(respones.text)
    print "get " + city + " weather data"


# 处理天气数据，多余删除，缺失的补上
def processing_weather(df, start_time="2017-01-01 00:00:00", end_time="2018-04-10 23:00:00"):
    # df = history_weather_data(city=city)
    start_day = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end_day = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
    dates = pd.date_range(start_day, end_day, freq='60min')
    df1 = pd.DataFrame(index=dates)
    df1['time'] = df1.index.map(lambda x: x)
    stations_group = df.groupby("station_id")
    ans = None
    for station_id, group in stations_group:
        # if station_id == "KF1":
        #     print "test"
        df1['station_id'] = np.array([station_id] * df1.values.shape[0])
        # if ans is None:
        #     print df1.values.shape
        #     print group
        group = pd.merge(df1, group, how='left')
        # if ans is None:
        #     print group
        if ans is None:
            ans = group
        else:
            ans = pd.concat([ans, group], axis=0)
            # print df1.values.shape
    # print ans
    ans = ans.fillna(method='pad')
    ans = ans.fillna(method='backfill')
    # print ans
    ans['time'] = pd.to_datetime(ans['time'])
    ans.index = ans['time']
    # filename = base_path_3 + city + '_weather_history.csv'
    # ans.to_csv(filename, index=False, sep=',')
    return ans


# 从网站下载数据
def get_weather_forecast_data(city):
    time_now = datetime.now()
    time_now = time_now - timedelta(hours=8)
    end_day = time_now.strftime('%Y-%m-%d')

    start_time = end_day + "-00"
    end_time = end_day + "-23"
    link3 = 'https://biendata.com/competition/meteorology/' + city + '_grid/' + start_time + '/' + end_time + '/2k0d1d8'
    respones = requests.get(link3)
    filename = base_path_2 + city + "_meteorology_grid_today.csv"
    with open(filename, 'w') as f:
        f.write(respones.text)
    df1 = pd.read_csv(filename, sep=',')
    df1['time'] = pd.to_datetime(df1['time'])
    df1.index = df1['time']
    max_hour = df1['time'].max().hour
    print max_hour
    start_time = end_day + '-%02d' % (max_hour-2)
    link1 = 'http://kdd.caiyunapp.com/competition/forecast/' + city + '/' + start_time + '/2k0d1d8'
    respones = requests.get(link1)
    filename2 = base_path_2 + city + "_meteorology_grid_forecast.csv"
    with open(filename2, 'w') as f:
        f.write(respones.text)
    df2 = pd.read_csv(filename2, sep=',')
    df2['time'] = pd.to_datetime(df2['forecast_time'])
    df2.index = df2['time']
    attr_need = ['time', 'station_id', 'temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction']
    df = pd.concat([df1[attr_need], df2[attr_need]])
    nearst_contrary = {
        'ld': {'london_grid_451': 'GB0', 'london_grid_452': 'RB7', 'london_grid_430': 'LW2', 'london_grid_409': 'CT3',
               'london_grid_408': 'CR8', 'london_grid_388': 'MY7', 'london_grid_366': 'TD5', 'london_grid_368': 'HR1',
               'london_grid_472': 'BX1', 'london_grid_346': 'LH0'},
        'bj': {'beijing_grid_282': 'fengtaihuayuan_aq', 'beijing_grid_283': 'wanliu_aq',
               'beijing_grid_385': 'yongledian_aq', 'beijing_grid_414': 'miyunshuiku_aq',
               'beijing_grid_239': 'yungang_aq', 'beijing_grid_238': 'fangshan_aq', 'beijing_grid_216': 'liulihe_aq',
               'beijing_grid_278': 'yufa_aq', 'beijing_grid_301': 'daxing_aq', 'beijing_grid_263': 'beibuxinqu_aq',
               'beijing_grid_368': 'shunyi_aq', 'beijing_grid_392': 'miyun_aq', 'beijing_grid_349': 'huairou_aq',
               'beijing_grid_452': 'donggaocun_aq', 'beijing_grid_224': 'badaling_aq', 'beijing_grid_225': 'yanqin_aq',
               'beijing_grid_264': 'pingchang_aq', 'beijing_grid_265': 'dingling_aq',
               'beijing_grid_304': 'aotizhongxin_aq', 'beijing_grid_303': 'wanshouxigong_aq',
               'beijing_grid_261': 'gucheng_aq', 'beijing_grid_262': 'zhiwuyuan_aq', 'beijing_grid_366': 'tongzhou_aq',
               'beijing_grid_324': 'nongzhanguan_aq', 'beijing_grid_240': 'mentougou_aq',
               'beijing_grid_323': 'yizhuang_aq'}}

    df = df[df["station_id"].isin(nearst_contrary[city].keys())]
    start_day = time_now + timedelta(days=2)
    start_day = start_day.strftime('%Y-%m-%d')
    start_time = end_day + " 00:00:00"
    end_time = start_day + " 23:00:00"
    ans = processing_weather(df, start_time, end_time)
    print ans
    ans.to_csv(base_path_2 + city + '_weather_forecast_caiyun.csv', index=False, sep=',')
    print "get " + city + " weather data"


def down_load(city='bj', start_day="2018-04-11"):
    # end_day = "2018-04-29"
    # end_time = end_day + "-23"
    start_time = start_day + "-0"
    get_data(start_time=start_time, end_time="", city=city, current_day=True)
    # start_time = "2018-04-28-0"
    get_weather_data(start_time=start_time, end_time="", city=city, current_day=True)


def loss_data_day(city):
    filename = base_path_2 + city + "_airquality_processing.csv"
    if city == 'bj':
        attr_need = ["PM25_Concentration", "PM10_Concentration", "O3_Concentration"]
    else:
        attr_need = ["PM25_Concentration", "PM10_Concentration"]
    df1 = pd.read_csv(filename, sep=',')
    df1['time'] = pd.to_datetime(df1['time'])
    df1.index = df1['time']
    df2 = pd.read_csv(base_path_2 + city + "_airquality_current_day.csv", sep=',')
    df2['time'] = pd.to_datetime(df2['time'])
    df2.index = df2['time']
    df2['time_week'] = df2.index.map(lambda x: x.weekday)
    df2['time_year'] = df2.index.map(lambda x: x.year)
    df2['time_month'] = df2.index.map(lambda x: x.month)
    df2['time_day'] = df2.index.map(lambda x: x.day)
    df2['time_hour'] = df2.index.map(lambda x: x.hour)
    df = pd.concat([df1, df2])
    groups = df.groupby(['station_id', 'time_year', 'time_month', 'time_day'])
    loss_rate = {}
    for (station_id, year, month, day), group in groups:
        value = group[attr_need].values
        if loss_rate.has_key(station_id) == False:
            loss_rate[station_id] = {}
        day = "%d-%02d-%02d" % (int(year), int(month), int(day))
        loss_rate[station_id][day] = {}
        rate_PM25 = np.isnan(value[:, 0]).sum() * 1.0 / (value.shape[0])
        rate_PM10 = np.isnan(value[:, 1]).sum() * 1.0 / (value.shape[0])
        loss_rate[station_id][day]["PM25"] = rate_PM25
        loss_rate[station_id][day]["PM10"] = rate_PM10
        if city == "bj":
            rate_O3 = np.isnan(value[:, 2]).sum() * 1.0 / (value.shape[0])
            loss_rate[station_id][day]["O3"] = rate_O3
        loss_rate[station_id][day]["all"] = np.isnan(value).sum() * 1.0 / (value.shape[0] * value.shape[1])
    return loss_rate
    # print(value)


def get_loss_rate():
    import cPickle as pickle
    filename = base_path_2 + "rate.pkl"
    f1 = file(filename, 'wb')
    loss_rate = {}
    loss_rate['bj'] = loss_data_day(city='bj')
    loss_rate['ld'] = loss_data_day(city='ld')
    pickle.dump(loss_rate, f1, True)


if __name__ == '__main__':
    time_now = datetime.now()
    time_now = time_now - timedelta(hours=8)
    start_day = (time_now - timedelta(days=2)).strftime('%Y-%m-%d')
    end_day = time_now.strftime('%Y-%m-%d')
    print start_day
    # start_day = '2018-05-03'
    # down_load('bj', start_day=start_day)
    # down_load('ld', start_day=start_day)
    # weather_data_forecast()

    # down_load('bj')
    # down_load('ld')

    # get_loss_rate()
    #
    get_weather_forecast_data(city='bj')
    get_weather_forecast_data(city='ld')
