# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import time
import re
import requests
import sys
from unit import *
from xgboost_with_weather import xgboost_run
from extraTree_with_weather import ext_with_weather_run
from ensemble import get_ans, get_ans_latter
from lightgbm_with_weather import lightgbm_run
from lightgbm_with_weather_log import lightgbm_log_run


# from down_load_data import down_load, weather_data_forecast
# from api_submit import submit

def run_later(start_day, end_day):
    xgboost_run(day1=start_day, day2=end_day, caiyun=False)
    ext_with_weather_run(day1=start_day, day2=end_day, caiyun=False)
    get_ans_latter(end_day=end_day, caiyun=False)

    xgboost_run(day1=start_day, day2=end_day, caiyun=True)
    ext_with_weather_run(day1=start_day, day2=end_day, caiyun=True)
    get_ans_latter(end_day=end_day, caiyun=True)

def run(start_day, end_day):
    lightgbm_run(day1=start_day, day2=end_day, caiyun=True)
    get_ans(end_day=end_day, caiyun=True)
    link3 = "http://axz.ciih.net/index.php/Api/Gsm/msg/msg/彩云数据算法结束"
    requests.get(link3)
    lightgbm_run(day1=start_day, day2=end_day, caiyun=False)
    get_ans(end_day=end_day, caiyun=False)

    lightgbm_log_run(day1=start_day, day2=end_day, caiyun=True)
    lightgbm_log_run(day1=start_day, day2=end_day, caiyun=False)
    pass


def main():
    link3 = "http://axz.ciih.net/index.php/Api/Gsm/msg/msg/算法开始"
    requests.get(link3)
    time_now = datetime.now()
    time_now = time_now - timedelta(hours=9)
    start_day = (time_now - timedelta(days=2)).strftime('%Y-%m-%d')
    end_day = time_now.strftime('%Y-%m-%d')
    # start_day = "2018-05-08"
    # end_day = "2018-05-10"
    run(start_day, end_day)
    link3 = "http://axz.ciih.net/index.php/Api/Gsm/msg/msg/算法结束"
    requests.get(link3)
    run_later(start_day, end_day)
    # filename = end_day + "lightgbm_mean_ensemble_29_6.csv"
    # submit(filename=filename)
    # link3 = "http://axz.ciih.net/index.php/Api/Gsm/msg/msg/提交成功1"
    # requests.get(link3)
    # filename = end_day + "lightgbm_ensemble_mean_weight.csv"
    # submit(filename=filename)
    # link3 = "http://axz.ciih.net/index.php/Api/Gsm/msg/msg/提交成功2"
    # requests.get(link3)
    # filename = end_day + "lightgbm_ensemble_median_2.csv"
    # submit(filename=filename)
    # link3 = "http://axz.ciih.net/index.php/Api/Gsm/msg/msg/提交成功3"
    # requests.get(link3)



if __name__ == '__main__':
    # main()
    while True:
        time_now = datetime.now()
        time_now = time_now  # - timedelta(hours=8)
        if time_now.hour == 6 and time_now.minute >= 20:
            main()
            time.sleep(60 * 60)
        else:
            time.sleep(60 * 2)
