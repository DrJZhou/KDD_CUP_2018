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
    lightgbm_run(day1=start_day, day2=end_day, caiyun=False)
    get_ans(end_day=end_day, caiyun=False)

    lightgbm_log_run(day1=start_day, day2=end_day, caiyun=True)
    lightgbm_log_run(day1=start_day, day2=end_day, caiyun=False)
    pass


def main():
    time_now = datetime.now()
    time_now = time_now - timedelta(hours=9)
    start_day = (time_now - timedelta(days=2)).strftime('%Y-%m-%d')
    end_day = time_now.strftime('%Y-%m-%d')
    # start_day = "2018-05-08"
    # end_day = "2018-05-10"
    run(start_day, end_day)
    run_later(start_day, end_day)



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
