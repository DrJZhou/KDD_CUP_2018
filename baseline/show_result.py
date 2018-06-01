# coding: utf-8
import pandas as pd
import numpy as np
import dateutil
import requests
import datetime
from matplotlib import pyplot as plt


def smape(actual, predicted):
    a = np.abs(np.array(actual) - np.array(predicted))
    b = np.array(actual) + np.array(predicted)

    return 2 * np.mean(np.divide(a, b, out=np.zeros_like(a), where=b != 0, casting='unsafe'))


from dateutil.parser import parse
from datetime import date, timedelta


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


utc_date = date_add_hours(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), -8)
print('现在是UTC时间：{}'.format(utc_date))
print('距离待预测时间还有{}个小时'.format(diff_of_hours(date_add_days(utc_date, 1), utc_date) + 1))

filename = "2018-05-31_ensemble_all_zhoujie"
day = "2018-05-31"
filepath = '../image/results/' + "".join(day.split("-")) + '/' + filename + '.csv'  # 0514

# filepath = './0513commit/bag_55.csv'   #0513
# filepath = './0513commit/api_concat_bag_55_6hours.csv'   #0513api

result = pd.read_csv(filepath)
now_date = day

api = False

# # 北京数据

# In[5]:


## 2018-04到最新的数据
if api:
    url = 'https://biendata.com/competition/airquality/bj/2018-04-25-0/2018-06-01-0/2k0d1d8'
    respones = requests.get(url)
    with open("../image/bj_aq_new_show.csv", 'w') as f:
        f.write(respones.text)

# In[6]:


replace_dict = {'wanshouxigong': 'wanshouxig', 'aotizhongxin': 'aotizhongx', 'nongzhanguan': 'nongzhangu',
                'fengtaihuayuan': 'fengtaihua',
                'miyunshuiku': 'miyunshuik', 'yongdingmennei': 'yongdingme', 'xizhimenbei': 'xizhimenbe'}

# In[7]:


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
bj_aq_new_show.head(2)

# # 伦敦数据

# In[8]:


## London 2018-04到最新的数据
if api:
    url = 'https://biendata.com/competition/airquality/ld/2018-03-30-23/2018-06-01-01/2k0d1d8'
    respones = requests.get(url)
    with open("../image/lw_aq_new.csv", 'w') as f:
        f.write(respones.text)

lw_aq_new = pd.read_csv('../image/lw_aq_new.csv')
lw_aq_new.columns = ['id', 'location', 'utc_time', 'PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
lw_aq_new = lw_aq_new[['utc_time', 'PM2.5', 'PM10', 'O3', 'location']]
lw_aq_new.head(2)

# In[9]:


aq_new = pd.concat([bj_aq_new_show, lw_aq_new])
aq_new['utc_time'] = pd.to_datetime(aq_new['utc_time'])
aq_new.head(3)

# In[10]:


bj_aq_new_show.utc_time.max()


# In[11]:


def getlocation(x):
    return x.split('_aq')[0].split('#')[0]


hour1 = pd.to_datetime('2018-01-06 01:00:00') - pd.to_datetime('2018-01-06 00:00:00')

result['location'] = result['test_id'].apply(lambda x: getlocation(x))
result['utc_time'] = result['test_id'].apply(lambda x: x.split('#')[1])
result['utc_time'] = result['utc_time'].apply(
    lambda x: str(pd.to_datetime(now_date + ' 00:00:00') + hour1 * (24 + int(x))))
result['utc_time'] = pd.to_datetime(result['utc_time'])
result.head(3)

# # figure

# In[12]:


# bj_aq_new_show.loc[bj_aq_new_show['location'] == 'fengtaihua',['utc_time','PM2.5']]


# In[13]:

from matplotlib.backends.backend_pdf import PdfPages
item_now = "PM2.5"
with PdfPages('../image/' + "bj_" + item_now + "_" + filename + '.pdf') as pdf:
    for location in bj_aq_new_show.location.unique():

        # for item in ['PM2.5','PM10','O3']:
        # for item in ['PM2.5']:
            #     for item in ['PM10']:
        for item in [item_now]:

            temp1 = bj_aq_new_show.loc[bj_aq_new_show['location'] == location, ['utc_time', item]]
            temp2 = result.loc[result['location'] == location, ['utc_time', item]]

            temp1 = temp1.sort_values(by='utc_time')
            #         temp2 = temp2.sort_values(by='utc_time')

            temp = pd.concat([temp1, temp2])
            temp['utc_time'] = pd.to_datetime(temp['utc_time'])
            #         temp = temp.sort_values(by='utc_time')
            temp.index = range(len(temp))
            start = temp.loc[0, 'utc_time']
            temp['t'] = (temp['utc_time'] - start) / hour1

            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(211)
            x = temp.loc[:len(temp) - 49, 't']
            y = temp.loc[:len(temp) - 49, item]

            x1 = temp.loc[len(temp) - 48:, 't']
            y1 = temp.loc[len(temp) - 48:, item]

            ax.plot(x, y, color='green')
            ax.plot(x1, y1, color='red')
            plt.title(location + item)
            pdf.savefig()
            plt.close()

# In[14]:

#
# if False:
#     for location in lw_aq_new.location.unique():
#
# #         for item in ['PM2.5','PM10']:
# #         for item in ['PM2.5']:
#         for item in ['PM10']:
#
#             temp1 = lw_aq_new.loc[lw_aq_new['location'] == location,['utc_time',item]]
#             temp2 = result.loc[result['location'] == location,['utc_time',item]]
#             temp = pd.concat([temp1,temp2])
#             temp['utc_time'] = pd.to_datetime(temp['utc_time'])
#             # temp = temp.sort_values(by='utc_time')
#             temp.index = range(len(temp))
#             start = temp.loc[0,'utc_time']
#             temp['t'] = (temp['utc_time'] - start)/hour1
#
#             fig = plt.figure(figsize=(5,5))
#             ax = fig.add_subplot(211)
#             x = temp.loc[:len(temp)-49,'t']
#             y = temp.loc[:len(temp)-49,item]
#
#             x1 = temp.loc[len(temp)-48:,'t']
#             y1 = temp.loc[len(temp)-48:,item]
#
#             ax.plot(x,y,color='green')
#             ax.plot(x1,y1,color='red')
#             plt.title(location+item)


# In[15]:


# pd.DataFrame(pd.date_range(temp.loc[0,'utc_time'],temp.loc[len(temp)-1,'utc_time'],freq='h'))
