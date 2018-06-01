# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
# import random
# from datetime import datetime, timedelta
# import time
# import re
# from sklearn.externals import joblib
# import requests
# import sys
# from unit import Distance1
# import xlrd
from unit import *
from crawl_data import *

reload(sys)
sys.setdefaultencoding('utf-8')
base_path_1 = "../dataset/"
base_path_2 = "../dataset/tmp/"
base_path_3 = "../output/"

nearst = {
    "bj": {'miyunshuiku_aq': 'beijing_grid_414', 'tiantan_aq': 'beijing_grid_303', 'yizhuang_aq': 'beijing_grid_323',
           'pingchang_aq': 'beijing_grid_264', 'zhiwuyuan_aq': 'beijing_grid_262', 'qianmen_aq': 'beijing_grid_303',
           'pinggu_aq': 'beijing_grid_452', 'beibuxinqu_aq': 'beijing_grid_263', 'shunyi_aq': 'beijing_grid_368',
           'tongzhou_aq': 'beijing_grid_366', 'yungang_aq': 'beijing_grid_239', 'yufa_aq': 'beijing_grid_278',
           'wanshouxigong_aq': 'beijing_grid_303', 'mentougou_aq': 'beijing_grid_240',
           'dingling_aq': 'beijing_grid_265',
           'donggaocun_aq': 'beijing_grid_452', 'nongzhanguan_aq': 'beijing_grid_324', 'liulihe_aq': 'beijing_grid_216',
           'xizhimenbei_aq': 'beijing_grid_283', 'fangshan_aq': 'beijing_grid_238', 'nansanhuan_aq': 'beijing_grid_303',
           'huairou_aq': 'beijing_grid_349', 'dongsi_aq': 'beijing_grid_303', 'badaling_aq': 'beijing_grid_224',
           'yanqin_aq': 'beijing_grid_225', 'gucheng_aq': 'beijing_grid_261', 'fengtaihuayuan_aq': 'beijing_grid_282',
           'wanliu_aq': 'beijing_grid_283', 'yongledian_aq': 'beijing_grid_385', 'aotizhongxin_aq': 'beijing_grid_304',
           'dongsihuan_aq': 'beijing_grid_324', 'daxing_aq': 'beijing_grid_301', 'miyun_aq': 'beijing_grid_392',
           'guanyuan_aq': 'beijing_grid_282', 'yongdingmennei_aq': 'beijing_grid_303'
           },
    "ld": {'KC1': 'london_grid_388', 'CD1': 'london_grid_388', 'HV1': 'london_grid_472', 'CD9': 'london_grid_409',
           'TD5': 'london_grid_366', 'GR4': 'london_grid_451', 'RB7': 'london_grid_452', 'TH4': 'london_grid_430',
           'HR1': 'london_grid_368', 'BL0': 'london_grid_409', 'GR9': 'london_grid_430', 'ST5': 'london_grid_408',
           'LH0': 'london_grid_346', 'KF1': 'london_grid_388', 'MY7': 'london_grid_388', 'BX9': 'london_grid_472',
           'GN3': 'london_grid_451', 'GN0': 'london_grid_451', 'CT2': 'london_grid_409', 'CT3': 'london_grid_409',
           'BX1': 'london_grid_472', 'CR8': 'london_grid_408', 'LW2': 'london_grid_430', 'GB0': 'london_grid_451'
           }
}

nearst_wounder_station = {
    'ld': {'london_grid_451': 'ILONDON1169', 'london_grid_452': 'IROMFORD7', 'london_grid_409': 'IGLAYIEW96',
           'london_grid_430': 'IGLAYIEW96', 'london_grid_408': 'ILONDONM4', 'london_grid_388': 'IGLAYIEW96',
           'london_grid_366': 'ILONDON633', 'london_grid_368': 'IGLAYIEW96', 'london_grid_472': 'IBEXLEY16',
           'london_grid_346': 'IGLAYIEW96'},
    'bj': {'beijing_grid_282': 'IBEIJING355', 'beijing_grid_283': 'I11HAIDI3', 'beijing_grid_385': 'ITONGZHO3',
           'beijing_grid_414': 'I11HOUSH2', 'beijing_grid_239': 'IBEIJING250', 'beijing_grid_238': 'IBEIJING250',
           'beijing_grid_216': 'IBEIJING250', 'beijing_grid_278': 'I11BAIZH2', 'beijing_grid_262': 'I11HAIDI3',
           'beijing_grid_263': 'ICHANGPI3', 'beijing_grid_324': 'ICHAOYAN11', 'beijing_grid_392': 'I11HOUSH2',
           'beijing_grid_349': 'I11HOUSH2', 'beijing_grid_452': 'ITONGZHO3', 'beijing_grid_224': 'ICHANGPI3',
           'beijing_grid_225': 'ICHANGPI3', 'beijing_grid_264': 'ICHANGPI3', 'beijing_grid_265': 'ICHANGPI3',
           'beijing_grid_304': 'IHAIDIAN9', 'beijing_grid_303': 'IXICHENG8', 'beijing_grid_261': 'IBEIJING250',
           'beijing_grid_301': 'I11BAIZH2', 'beijing_grid_366': 'ITONGZHO3', 'beijing_grid_368': 'I11HOUSH2',
           'beijing_grid_240': 'IBEIJING250', 'beijing_grid_323': 'I11BAIZH2'}}


# 从网站下载数据
def get_data(city, start_time, end_time, current_day=False):
    # if city == "bj":
    #     link2 = 'https://biendata.com/competition/meteorology/' + city + '/' + start_time + '/' + end_time + '/2k0d1d8'
    #     respones = requests.get(link2)
    #     if current_day == False:
    #         with open(base_path_2 + city + "_meteorology_" + start_time + "_" + end_time + ".csv", 'w') as f:
    #             f.write(respones.text)
    #     else:
    #         with open(base_path_2 + city + "_meteorology_current_day.csv", 'w') as f:
    #             f.write(respones.text)
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


# 加载最近的网格站点
def load_nearst_contrary():
    nearst_contrary = {}
    for city in nearst.keys():
        nearst_contrary[city] = {}
        for station1, station2 in nearst[city].items():
            nearst_contrary[city][station2] = station1
    return nearst_contrary


# 加载 历史数据 2018年3月27号之前
def load_data(city, flag=False):
    if city == "bj":
        filename = base_path_1 + "Beijing_historical_meo_grid.csv"
    else:
        filename = base_path_1 + "London_historical_meo_grid.csv"
    df = pd.read_csv(filename, sep=',')
    nearst_contrary = load_nearst_contrary()
    df = df[df["stationName"].isin(nearst_contrary[city].keys())]
    # df = df["stationName"].replace(nearst_contrary[city].keys(), nearst[city].keys())
    df.rename(columns={'stationName': 'station_id', 'utc_time': 'time', 'wind_speed/kph': 'wind_speed'}, inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    df.index = df['time']
    attr_need = ['station_id', 'time', 'temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed']
    if flag == True:
        df = df[attr_need]
    # print df
    return df[: "2018-03-26"]


# 加载api获得的数据
def load_data_api(city, current_flag=False):
    if current_flag == True:
        filename = base_path_2 + city + "_meteorology_grid_current_day.csv"
    else:
        filename = base_path_2 + city + "_meteorology_grid_2018-04-01-0_2018-04-10-23.csv"
    df = pd.read_csv(filename, sep=',')
    # nearst_contrary = load_nearst_contrary()
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
    df['time'] = pd.to_datetime(df['time'])
    df.index = df['time']
    attr_need = ['station_id', 'time', 'temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed']
    df = df[attr_need]
    return df


# 加载wunder网站上站点信息
def load_station_in_wounder():
    station_wounder = {}
    station_wounder['bj'] = {}
    filename = base_path_1 + "station_beijing.txt"
    for line in open(filename).readlines():
        data = line.strip().split(",")
        station_wounder['bj'][data[0]] = {"lat": float(data[1]), "lng": float(data[2])}

    station_wounder['ld'] = {}
    filename = base_path_1 + "station_london.txt"
    for line in open(filename).readlines():
        data = line.strip().split(",")
        station_wounder['ld'][data[0]] = {"lat": float(data[1]), "lng": float(data[2])}
    return station_wounder


# 加载grid站点信息
def load_grid_stations():
    grid_stations = {}
    df1 = load_data(city="bj")
    values = df1[["station_id", "latitude", "longitude"]].drop_duplicates().values
    grid_stations['bj'] = {}
    for i in range(values.shape[0]):
        grid_stations['bj'][values[i, 0]] = {"lat": values[i, 1], "lng": values[i, 2]}

    df2 = load_data(city='ld')
    values = df2[["station_id", "latitude", "longitude"]].drop_duplicates().values
    grid_stations['ld'] = {}
    for i in range(values.shape[0]):
        grid_stations['ld'][values[i, 0]] = {"lat": values[i, 1], "lng": values[i, 2]}
    return grid_stations


# 获取最近的wunder网站上的站点
def get_nearst_wounder_station():
    grids_station = load_grid_stations()
    wounder_station = load_station_in_wounder()
    nearst_grid_wounder = {}
    for city in grids_station.keys():
        nearst_grid_wounder[city] = {}
        for station_id in grids_station[city].keys():
            lat_1 = grids_station[city][station_id]['lat']
            lng_1 = grids_station[city][station_id]['lng']
            min_distance = None
            min_station = None
            for station_id_2 in wounder_station[city].keys():
                lat_2 = wounder_station[city][station_id_2]['lat']
                lng_2 = wounder_station[city][station_id_2]['lng']
                distance = Distance1(lat_1, lng_1, lat_2, lng_2)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    min_station = station_id_2
            nearst_grid_wounder[city][station_id] = min_station
    return nearst_grid_wounder


# 抓取0327_0331的数据
def data_from_0327_0331():
    for city in nearst_wounder_station.keys():
        filename = base_path_2 + city + "_0327_0331.csv"
        fr = open(filename, 'wb')
        fr.write("station_id,time,temperature,humidity,pressure,wind_speed,wind_direction\n")
        for grid_station in nearst_wounder_station[city].keys():
            wounder_staion = nearst_wounder_station[city][grid_station]
            for day in range(27, 33):
                if day == 32:
                    data = weather_station(year=2018, month=4, day=1, station_id=wounder_staion)
                else:
                    data = weather_station(year=2018, month=3, day=day, station_id=wounder_staion)
                data1 = data["history"]["observations"]
                for j in range(24):
                    min_time = 10000
                    tmp = None
                    if day == 32:
                        date_time = "2018-04-01 %02d:00:00" % j
                    else:
                        date_time = "2018-03-%02d %02d:00:00" % (day, j)
                    for i in range(len(data1)):
                        date = data1[i]['date']['iso8601'].split("+")[0].replace("T", " ")
                        # print date
                        date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                        hour = date.hour
                        minu = date.minute
                        # if hour > j:
                        #     break
                        # if hour < j - 1:
                        #     continue
                        temperature = data1[i]['temperature']
                        pressure = data1[i]['pressure']
                        humidity = data1[i]['humidity']
                        wind_speed = data1[i]['wind_speed']
                        wind_dir = data1[i]['wind_dir']
                        if hour >= j and minu + (hour - j) * 60 < min_time:
                            min_time = minu + (hour - j) * 60
                            tmp = [grid_station, date_time, temperature, humidity, pressure, wind_speed, wind_dir]
                        if hour < j and (j - hour) * 60 - minu < min_time:
                            min_time = (j - hour) * 60 - minu
                            tmp = [grid_station, date_time, temperature, humidity, pressure, wind_speed, wind_dir]
                    fr.write(",".join([str(x) for x in tmp]) + "\n")
        fr.close()


# 处理抓取的0327-0331的数据
def process_crawl_data(city="bj"):
    filename = base_path_2 + city + "_0327_0331.csv"
    df = pd.read_csv(filename, sep=',')
    df['time'] = pd.to_datetime(df['time'])
    if city == "bj":
        df['time'] = df['time'] - timedelta(hours=8)
    else:
        df['time'] = df['time'] - timedelta(hours=1)
    df.index = df['time']
    return df["2018-03-27":"2018-03-31"]


# 将除了0410以后的数据放一起
def load_data_city(city="bj"):
    grids = load_data(city=city, flag=True)
    crawl_data = process_crawl_data(city=city)
    df = pd.concat([grids, crawl_data])
    # print df
    data_0401_0410 = load_data_api(city=city, current_flag=False)
    df = pd.concat([df, data_0401_0410])
    df = df.sort_index()
    df = processing_weather(df, start_time="2017-01-01 00:00:00", end_time="2018-04-10 23:00:00")
    print df.values.shape
    df.to_csv(base_path_3 + city + '_weather_history.csv', index=False, sep=',')
    return df


# 最近数据加载
def post_weather_data(city):
    # df = load_data_api(city=city, current_flag=True)
    # df = processing_weather(df, start_time="2018-04-11 00:00:00", end_time="2018-04-17 23:00:00")
    # print df
    # df.to_csv(base_path_3 + city + '_weather_post.csv', index=False, sep=',')
    filename = base_path_3 + city + '_weather_post.csv'
    df = pd.read_csv(filename, sep=',')
    df['time'] = pd.to_datetime(df['time'])
    df.index = df['time']
    return df


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


# 历史天气数据 "2018-04-10"之前都有
def history_weather_data(city, start_day="2017-01-01", end_day="2018-04-10"):
    # df = load_data_api(city=city, current_flag=True)
    # df.to_csv(base_path_3 + city + '_weather_post.csv', index=False, sep=',')
    filename = base_path_3 + city + '_weather_history.csv'
    df = pd.read_csv(filename, sep=',')
    df['time'] = pd.to_datetime(df['time'])
    df.index = df['time']
    df = df[start_day:end_day]
    return df


# 抓取天气预测数据
def weather_data_forecast():
    grid_stations = load_grid_stations()
    # wounder_staions = load_station_in_wounder()
    for city in nearst_wounder_station.keys():
        filename = base_path_2 + city + "_weather_forecast.csv"
        fr = open(filename, 'wb')
        fr.write("station_id,time,temperature,humidity,pressure,wind_speed,wind_direction\n")
        for grid_station in nearst_wounder_station[city].keys():
            # wounder_staion = nearst_wounder_station[city][grid_station]
            lat = grid_stations[city][grid_station]['lat']
            lng = grid_stations[city][grid_station]['lng']
            data = weather_lat_lng(lat, lng)
            data1 = data['history_hours']
            data2 = data["forecasts_hours"]
            for i in range(len(data1)):
                date = data1[i]['date']['iso8601'].split("+")[0].replace("T", " ")
                # print date
                if city == "bj":
                    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S") - timedelta(hours=8)
                else:
                    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S") - timedelta(hours=1)
                temperature = (data1[i]['temperature'] - 32.0) / 1.8
                pressure = data1[i]['pressure'] / 295.2998751 * 10000
                humidity = data1[i]['humidity']
                wind_speed = data1[i]['wind_speed']
                wind_dir = data1[i]['wind_dir']
                tmp = [grid_station, date, temperature, humidity, pressure, wind_speed, wind_dir]
                fr.write(",".join([str(x) for x in tmp]) + "\n")
            for i in range(len(data2)):
                date = data2[i]['fcst_valid_local'].split("+")[0].replace("T", " ")
                # print date
                if city == "bj":
                    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S") - timedelta(hours=8)
                else:
                    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S") - timedelta(hours=1)
                temperature = (data2[i]['temp'] - 32.0) / 1.8
                pressure = data2[i]['mslp'] / 295.2998751 * 10000
                humidity = data2[i]['rh']
                wind_speed = data2[i]['wspd']
                wind_dir = data2[i]['wdir']
                tmp = [grid_station, date, temperature, humidity, pressure, wind_speed, wind_dir]
                fr.write(",".join([str(x) for x in tmp]) + "\n")
        fr.close()
    # https://openweathermap.org/data/2.5/forecast/?appid=b6907d289e10d714a6e88b30761fae22&id=1816670&units=metric


# 加载天气预测数据
def load_weather_forecast_data(city, caiyun=False):
    if caiyun == False:
        filename = base_path_2 + city + "_weather_forecast.csv"
    else:
        filename = base_path_2 + city + "_weather_forecast_caiyun.csv"
    df = pd.read_csv(filename, sep=',')
    df['time'] = pd.to_datetime(df['time'])
    df.index = df['time']
    time_now = datetime.now()
    time_now = time_now - timedelta(hours=8)
    start_day = time_now.strftime('%Y-%m-%d')
    end_day = datetime_toString(string_toDatetime(start_day) + timedelta(days=2))
    df = df[start_day:end_day]
    df = processing_weather(df, start_time=start_day + " 00:00:00", end_time=end_day + " 23:00:00")
    # print df
    return df


# # 加载4月10号以后的数据
# def load_all_weather_data(city, start_day="2018-04-11", end_day="2018-04-18", crawl_data=False):
#     # df1 = history_weather_data(city=city)
#     df2 = post_weather_data(city=city)
#     # print df2
#     one_day_before_start_day = datetime_toString(string_toDatetime(start_day) - timedelta(days=1))
#     df2 = df2[: one_day_before_start_day]
#     # df = pd.concat([df1, df2])
#     if crawl_data:
#         get_data(city, start_day + "-0", end_day + "-23", current_day=True)
#     df3 = load_data_api(city=city, current_flag=True)
#     one_day_before_end_day = datetime_toString(string_toDatetime(end_day) - timedelta(days=1))
#     one_day_after_end_day = datetime_toString(string_toDatetime(end_day) + timedelta(days=1))
#     two_day_after_end_day = datetime_toString(string_toDatetime(end_day) + timedelta(days=2))
#
#     time_now = datetime.now()
#     time_now = time_now - timedelta(hours=8)
#     time_now = time_now.strftime('%Y-%m-%d')
#     if string_toDatetime(two_day_after_end_day) < string_toDatetime(time_now):
#         df3 = processing_weather(df3, start_time=start_day + " 00:00:00", end_time=two_day_after_end_day + " 23:00:00")
#     elif string_toDatetime(one_day_after_end_day) < string_toDatetime(time_now):
#         df3 = processing_weather(df3, start_time=start_day + " 00:00:00", end_time=one_day_after_end_day + " 23:00:00")
#     elif string_toDatetime(one_day_after_end_day) == string_toDatetime(time_now):
#         df3 = processing_weather(df3, start_time=start_day + " 00:00:00", end_time=end_day + " 23:00:00")
#     else:
#         df3 = processing_weather(df3, start_time=start_day + " 00:00:00", end_time=one_day_before_end_day + " 23:00:00")
#     # print df3
#
#     df_post = pd.concat([df2, df3])
#     df_post.to_csv(base_path_3 + city + '_weather_post.csv', index=False, sep=',')
#     # print df3
#
#     df = pd.concat([df2, df3])
#
#     if crawl_data == True:
#         weather_data_forecast()
#     df4 = load_weather_forecast_data(city)
#     df = pd.concat([df, df4])
#     return df[: two_day_after_end_day]

# 加载4月10号以后的数据
def load_all_weather_data(city, start_day="2018-04-11", end_day="2018-04-18", crawl_data=False, caiyun=False):
    # df1 = history_weather_data(city=city)
    df2 = post_weather_data(city=city)
    # df2 = df2.drop_duplicates()
    max_post_day = datetime_toString(df2['time'].max() - timedelta(hours=23))
    print max_post_day
    # print df2
    one_day_after_max_post_day = datetime_toString(string_toDatetime(max_post_day) + timedelta(days=1))
    df2 = df2[: max_post_day]
    # df = pd.concat([df1, df2])
    if crawl_data:
        get_data(city, start_day + "-0", end_day + "-23", current_day=True)
    df3 = load_data_api(city=city, current_flag=True)
    one_day_before_end_day = datetime_toString(string_toDatetime(end_day) - timedelta(days=1))
    one_day_after_end_day = datetime_toString(string_toDatetime(end_day) + timedelta(days=1))
    two_day_after_end_day = datetime_toString(string_toDatetime(end_day) + timedelta(days=2))

    time_now = datetime.now()
    time_now = time_now - timedelta(hours=8)  # 8
    print time_now, start_day, end_day
    time_now = time_now.strftime('%Y-%m-%d')
    if string_toDatetime(two_day_after_end_day) < string_toDatetime(time_now):
        df3 = processing_weather(df3, start_time=start_day + " 00:00:00", end_time=two_day_after_end_day + " 23:00:00")
    elif string_toDatetime(one_day_after_end_day) < string_toDatetime(time_now):
        df3 = processing_weather(df3, start_time=start_day + " 00:00:00", end_time=one_day_after_end_day + " 23:00:00")
    elif string_toDatetime(one_day_after_end_day) == string_toDatetime(time_now):
        df3 = processing_weather(df3, start_time=start_day + " 00:00:00", end_time=end_day + " 23:00:00")
    else:
        df3 = processing_weather(df3, start_time=start_day + " 00:00:00", end_time=one_day_before_end_day + " 23:00:00")
    # print "-------------------------------------------\n", df3
    print one_day_after_max_post_day
    # print df2
    # print df3
    # print df3[one_day_after_max_post_day:]
    df_post = pd.concat([df2, df3[one_day_after_max_post_day:]])
    # print df_post
    # df_post = df_post.drop_duplicates()
    df_post.to_csv(base_path_3 + city + '_weather_post.csv', index=False, sep=',')
    # print df3

    max_post_day_new = datetime_toString(df_post['time'].max() - timedelta(hours=23))
    print max_post_day_new
    one_day_after_max_post_day_new = datetime_toString(string_toDatetime(max_post_day_new) + timedelta(days=1))

    if crawl_data == True:
        weather_data_forecast()
    df4 = load_weather_forecast_data(city, caiyun=caiyun)
    df = pd.concat([df_post, df4[one_day_after_max_post_day_new:]])
    # if string_toDatetime(two_day_after_end_day) > string_toDatetime(time_now):
    #     df = pd.concat([df_post, df4])
    # else:
    #     df = df_post
    # df = df.drop_duplicates()
    return df[: two_day_after_end_day]


# 最近数据加载
def post_weather_data_update(city):
    filename = base_path_2 + city + "_meteorology_grid_2018-04-11-0_2018-04-29-23.csv"
    df = pd.read_csv(filename, sep=',')
    nearst_contrary = load_nearst_contrary()
    df = df[df["station_id"].isin(nearst_contrary[city].keys())]
    df['time'] = pd.to_datetime(df['time'])
    df.index = df['time']
    attr_need = ['station_id', 'time', 'temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed']
    df = df[attr_need]
    df = processing_weather(df, start_time="2018-04-11 00:00:00", end_time="2018-04-29 23:00:00")
    print df
    df.to_csv(base_path_3 + city + '_weather_post.csv', index=False, sep=',')


if __name__ == '__main__':
    # grids = load_data(city='bj')
    # grids = load_data(city="ld")
    # nearst_wounder_station = get_nearst_wounder_station()
    # print nearst_wounder_station
    # data_from_0327_0331()
    # process_crawl_data(city='bj')
    # get_data(city="bj", start_time="2018-04-11-0", end_time="2018-04-18-23", current_day=True)
    # get_data(city="ld", start_time="2018-04-11-0", end_time="2018-04-18-23", current_day=True)
    # load_data_city(city='bj')
    # load_data_city(city='ld')
    # post_weather_data(city='bj')
    # post_weather_data(city='ld')
    # processing_weather(city='bj', start_time="2017-01-01 00:00:00", end_time="2018-04-10 23:00:00")
    # processing_weather(city='ld', start_time="2017-01-01 00:00:00", end_time="2018-04-10 23:00:00")
    # data = history_weather_data(city='bj', start_day="2017-01-01", end_day="2018-04-10")
    #
    # groups = data.groupby("station_id")
    # for station, group in groups:
    #     print group.values.shape
    # print "test"
    # data_0401_0410 = load_data_api(city='bj', current_flag=False)
    # groups = data_0401_0410.groupby("station_id")
    # for station, group in groups:
    #     print group.values.shape
    # load_all_data(city='bj')
    load_data_api(city='bj', current_flag=True)
    # weather_data_forecast()
    # load_weather_forecast_data(city='bj')
    # load_weather_forecast_data(city='ld')
    # weather_history = history_weather_data(city='bj', start_day="2018-01-01", end_day="2018-04-10")
    # # print weather_history
    # df2 = load_all_weather_data(city='bj', start_day="2018-04-17", end_day="2018-04-19", crawl_data=False)
    # print df2
    # df = pd.concat([weather_history, df2])
    # groups = df.groupby("station_id")
    # for station, group in groups:
    #     print group.values.shape
    # pass
    # post_weather_data_update(city='bj')
    # post_weather_data_update(city='ld')
