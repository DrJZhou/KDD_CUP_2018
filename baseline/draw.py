# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
import time
import sys

def draw_single_station_day(df, city, stations, start_day='2017-01-01', num=3):
    start_day = datetime.strptime(start_day, '%Y-%m-%d')
    from matplotlib.backends.backend_pdf import PdfPages

    for station_id in stations[city].keys():
        with PdfPages('../image/' + city + '_' + station_id + '.pdf') as pdf:
            for i in range(num):
                day = start_day + timedelta(days=i)
                day = day.strftime('%Y-%m-%d')
                df_ans = df[day]
                df_ans = df_ans[(df_ans.station_id==station_id)]
                if city == 'bj':
                    df_ans = df_ans[["station_id", "PM25_Concentration", "PM10_Concentration", "O3_Concentration"]]
                else:
                    df_ans = df_ans[["station_id", "PM25_Concentration", "PM10_Concentration"]]
                if df_ans.size == 0:

                    continue
                plt.figure(figsize=(12, 7))
                df_ans.plot()
                # print df_ans
                plt.title('station_id:' + station_id + '   ' + day)
                pdf.savefig()
                plt.close()


def draw_single_station(df, city, stations, start_day='2017-01-01'):
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages('../image/' + city + "_" + start_day + '_all.pdf') as pdf:
        for station_id in stations[city].keys():
            df_ans = df[start_day:]
            df_ans = df_ans[(df_ans.station_id==station_id)]
            if city == 'bj':
                df_ans = df_ans[["station_id", "PM25_Concentration", "PM10_Concentration", "O3_Concentration"]]
            else:
                df_ans = df_ans[["station_id", "PM25_Concentration", "PM10_Concentration"]]
            if df_ans.size == 0:

                continue
            plt.figure(figsize=(12, 7))
            df_ans.plot(subplots=True)
            # print df_ans
            plt.title('station_id:' + station_id)
            pdf.savefig()
            plt.close()


