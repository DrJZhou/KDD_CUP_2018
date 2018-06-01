# coding: utf-8
import pandas as pd
import numpy as np
import dateutil
import requests
import datetime
from matplotlib import pyplot as plt
from dateutil.parser import parse
from datetime import timedelta
import os
# from tqdm import tqdm

import time


# time.sleep(3600*6)

def smape(actual, predicted):
    a = np.abs(np.array(actual) - np.array(predicted))
    b = np.array(actual) + np.array(predicted)

    return 2 * np.mean(np.divide(a, b, out=np.zeros_like(a), where=b != 0, casting='unsafe'))


def date_add_hours(start_date, hours):
    end_date = parse(start_date) + timedelta(hours=hours)
    end_date = end_date.strftime('%Y-%m-%d %H:%M:%S')
    return end_date


def date_add_days(start_date, days):
    end_date = parse(start_date[:10]) + timedelta(days=days)
    end_date = end_date.strftime('%Y-%m-%d')
    return end_date


def diff_of_hours(time1, time2):
    hours = (parse(time1) - parse(time2)).total_seconds() // 3600
    return abs(hours)


def getlocation(x):
    return x.split('_aq')[0].split('#')[0]


def xishu(x, k, location='beijing'):
    if location == 'beijing':
        if len(x[0].split('#')[0]) > 3:
            return x[1] * k
        else:
            return x[1]
    if location == 'london':
        if len(x[0].split('#')[0]) == 3:
            return x[1] * k
        else:
            return x[1]


hour1 = pd.to_datetime('2018-01-06 01:00:00') - pd.to_datetime('2018-01-06 00:00:00')

utc_date = date_add_hours(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), -8)
print('现在是UTC时间：{}'.format(utc_date))
print('距离待预测时间还有{}个小时'.format(diff_of_hours(date_add_days(utc_date, 1), utc_date) + 1))

load_data = False

## 2018-04到最新的数据
if load_data:
    url = 'https://biendata.com/competition/airquality/bj/2018-05-10-0/2018-06-05-0/2k0d1d8'
    respones = requests.get(url)
    with open("../image/bj_aq_new_show.csv", 'w') as f:
        f.write(respones.text)

replace_dict = {'wanshouxigong': 'wanshouxig', 'aotizhongxin': 'aotizhongx', 'nongzhanguan': 'nongzhangu',
                'fengtaihuayuan': 'fengtaihua',
                'miyunshuiku': 'miyunshuik', 'yongdingmennei': 'yongdingme', 'xizhimenbei': 'xizhimenbe'}

bj_aq_new_show = pd.read_csv('../image/bj_aq_new_show.csv')
bj_aq_new_show.columns = ['id', 'stationId', 'utc_time', 'PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
bj_aq_new_show = bj_aq_new_show[['stationId', 'utc_time', 'PM2.5', 'PM10', 'O3']]

bj_aq_new_show['location'] = bj_aq_new_show['stationId'].apply(lambda x: x.split('_aq')[0])
bj_aq_new_show['location'] = bj_aq_new_show['location'].replace(replace_dict)
# bj_aq_new_show['utc_time'] = pd.to_datetime(bj_aq_new_show['utc_time'])
bj_aq_new_show = bj_aq_new_show[['utc_time', 'PM2.5', 'PM10', 'O3', 'location']]
# bj_aq_new_show.head(2)

# load_data = True
## London 2018-04到最新的数据
if load_data:
    url = 'https://biendata.com/competition/airquality/ld/2018-05-10-23/2018-06-05-01/2k0d1d8'
    respones = requests.get(url)
    with open("../image/lw_aq_new.csv", 'w') as f:
        f.write(respones.text)

lw_aq_new = pd.read_csv('../image/lw_aq_new.csv')
lw_aq_new.columns = ['id', 'location', 'utc_time', 'PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
lw_aq_new = lw_aq_new[['utc_time', 'PM2.5', 'PM10', 'O3', 'location']]
# lw_aq_new.head(2)

aq_new = pd.concat([bj_aq_new_show, lw_aq_new])
aq_new['utc_time'] = pd.to_datetime(aq_new['utc_time'])

# now_date = utc_date[:10]
day = '2018-05-30'
path = "../image/results/" + "".join(day.split("-")) + '/'
rule = False
names = [na for na in os.listdir(path) if na.endswith('.csv')]

show = []
# 历史数据手动设置：
now_date = day

# 0519
submit = ['2018-05-30-lightgbm_weather_params_5_2017_0101-2018_0515_less_False.csv',
          '2018-05-30lightgbm_ensemble_mean_4.csv',
          'piu_091_2018-05-30lightgbm_ensemble_median_4_009_2.csv',
          'piu_091_2018-05-30lightgbm_ensemble_median_4_009.csv']
# 0520
# submit = ['2018-05-31_ensemble_all_zhoujie.csv',
#           'piu_091_2018-05-31lightgbm_ensemble_median_4_009.csv',
#           '2018-05-31lightgbm_ensemble_mean_4.csv']

# for fi in range(len(names)):
for fi in submit:
    try:
        name = names[fi]
    except:
        name = fi

    result = pd.read_csv(path + name)

    result['location'] = result['test_id'].apply(lambda x: getlocation(x))
    result['utc_time'] = result['test_id'].apply(lambda x: x.split('#')[1])
    result['utc_time'] = result['utc_time'].apply(
        lambda x: str(pd.to_datetime(now_date + ' 00:00:00') + hour1 * (24 + int(x))))
    result['utc_time'] = pd.to_datetime(result['utc_time'])
    # print(result['location'])

    # # Rule
    item = 'PM2.5'
    weizhi = 'beijing'

    for k in range(0, 1):
        if rule:
            print(weizhi + ',' + item + ': ' + str(1 + 0.01 * k))

        result_rule = result.copy()
        #     result_rule['O3'] = result_rule['O3'].apply(lambda x:x*(1+0.01*k))

        # 用于寻找参数 或 计算线下得分
        result_rule[item] = result_rule[['test_id', item]].apply(lambda x: xishu(x, 1 + 0.01 * k, weizhi), axis=1)

        # 用于计算全部参数的效果
        # result_rule['O3'] = result_rule[['test_id','O3']].apply(lambda x:xishu(x,1.03,'beijing'),axis=1)
        # result_rule['PM2.5'] = result_rule[['test_id','PM2.5']].apply(lambda x:xishu(x,1.10,'beijing'),axis=1)
        # result_rule['PM10'] = result_rule[['test_id','PM10']].apply(lambda x:xishu(x,1.04,'beijing'),axis=1)

        res = []
        res_rule = []
        for location in result.location.unique():

            res_temp = [location]
            res_temp_rule = [location]

            temp1 = aq_new.loc[aq_new['location'] == location, ['utc_time', 'PM2.5', 'PM10', 'O3']]
            temp2 = result.loc[result['location'] == location, ['utc_time', 'PM2.5', 'PM10', 'O3']]
            temp3 = result_rule.loc[result_rule['location'] == location, ['utc_time', 'PM2.5', 'PM10', 'O3']]

            start_time = pd.to_datetime(now_date + ' 00:00:00') + hour1 * 24

            #         获得部分数据：
            #         end_time = temp1['utc_time'].max()

            #         获得全部数据：
            end_time = temp3['utc_time'].max()
            # end_time = temp3['utc_time'].max() - hour1 * (24+15)

            for i in range(int((end_time - start_time) / hour1) + 1):
                # for i in range(int((end_time - start_time)/hour1)+1):

                if location in ['CD1', 'BL0', 'GR4', 'MY7', 'HV1', 'GN3', 'GR9', 'LW2', 'GN0', 'KF1', 'CD9', 'ST5',
                                'TH4']:
                    actual = temp1.loc[temp1['utc_time'] == start_time + hour1 * i, ['PM2.5', 'PM10']].values
                    predicted = temp2.loc[temp2['utc_time'] == start_time + hour1 * i, ['PM2.5', 'PM10']].values
                    predicted_rule = temp3.loc[temp3['utc_time'] == start_time + hour1 * i, ['PM2.5', 'PM10']].values

                else:
                    actual = temp1.loc[temp1['utc_time'] == start_time + hour1 * i, ['PM2.5', 'PM10', 'O3']].values
                    predicted = temp2.loc[temp2['utc_time'] == start_time + hour1 * i, ['PM2.5', 'PM10', 'O3']].values
                    predicted_rule = temp3.loc[
                        temp3['utc_time'] == start_time + hour1 * i, ['PM2.5', 'PM10', 'O3']].values

                # print(smape(actual,predicted))

                res_temp.append(smape(actual, predicted))
                res_temp_rule.append(smape(actual, predicted_rule))
            #     print('------')
            res.append(res_temp)
            res_rule.append(res_temp_rule)
        #         print(res_temp)
        #         print(res_temp_rule)

        res = pd.DataFrame(res)
        # print(res)
        res_rule = pd.DataFrame(res_rule)

        origin_low = 'origin low:' + str((np.nanmean(res.iloc[:35,1:]) + np.nanmean(res.iloc[35:,1:]))/2)

        # origin_low = 'origin low:' + str(np.nanmean(res.iloc[35:, 1:]))

        origin_up = ';  origin up:' + str((np.nanmean(res.iloc[:35, 1:]) * 3 + np.nanmean(res.iloc[35:, 1:]) * 2) / 5)

        origin_mean = 'origin mean:' + str(
            ((np.nanmean(res.iloc[:35, 1:]) + np.nanmean(res.iloc[35:, 1:])) / 2) * 0.5 + (
                        (np.nanmean(res.iloc[:35, 1:]) * 3 + np.nanmean(res.iloc[35:, 1:]) * 2) / 5) * 0.5)

        rule_low = '  rule low:' + str((np.nanmean(res_rule.iloc[:35, 1:]) + np.nanmean(res_rule.iloc[35:, 1:])) / 2)
        rule_up = ';  rule up:' + str(
            (np.nanmean(res_rule.iloc[:35, 1:]) * 3 + np.nanmean(res_rule.iloc[35:, 1:]) * 2) / 5)

        rule_mean = ';  rule mean:' + str(
            ((np.nanmean(res_rule.iloc[:35, 1:]) + np.nanmean(res_rule.iloc[35:, 1:])) / 2) * 0.5 + (
                        (np.nanmean(res_rule.iloc[:35, 1:]) * 3 + np.nanmean(res_rule.iloc[35:, 1:]) * 2) / 5) * 0.5)

        if rule:
            print(name)
            # print(origin_low+origin_up)
            # print(rule_low+rule_up)
            # print(start_time, end_time)
            print(origin_mean + rule_mean)

    if not rule:
        # print(origin_low + origin_up + ';  ' + origin_mean)
        # print(name+': '+origin_mean[12:])
        print(name + ': ' + origin_low[11:])
        show.append([name, origin_low[11:]])

show = pd.DataFrame(show)
show.columns = ['file_name', 'smape']
show = show.sort_values(by='smape')
show['submit'] = 0
show.loc[show['file_name'].isin(submit), 'submit'] = 1

print(show)
show.to_csv('../image/' + day + '_show.csv', index=False, header=False)

print('预测时间段: ' + str(start_time) + '  ' + str(end_time))
print('缺失数据量: {}'.format(end_time - temp1['utc_time'].max()))

# res.to_csv('./0501smape.csv',index=False)
