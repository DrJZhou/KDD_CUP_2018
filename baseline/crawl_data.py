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
import json

reload(sys)
sys.setdefaultencoding('utf-8')

# 抓取城市的数据，每小时的数据，以及一天的数据
# https://www.wunderground.com/history/airport/EGMC/2018/4/4/DailyHistory.html?req_city=London&req_statename=England
'''
ans{
	"hours_data": [[time,temp,windchill,dew_point,humidity,pressure,visiability,wind_dir,wind_speed,gust_speed,precip,events,conditions], [], ..., []]
	"day_data":[temp_avg,temp_max,temp_min,degree_day,dew_point,humidity_avg,humidity_max,humidity_min,rain_value,pressure,speed_avg,speed_max,visiability,events]
}
hours_data: 时间 (BST)	气温	风冷温	露点	湿度	气压	    能见度	    Wind Dir	风速	                    瞬间风速	Precip	活动	状况
样例：      12:20 AM	5.0 °C	2.5 °C	3.0 °C	87%	    1010 百帕	7.0 千米	东南偏南	11.1 公里/小时 / 3.1 m/s	-	        N/A	    中雨	小雨
day_data:   平均温度	最高温度	最低温度	采温度日数	露点	平均湿度	最高湿度	最小湿度	降水量	气压	风速	最快风速	能见度	活动
'''


def city_weather(year=2018, month=4, day=7, city="beijing"):
    ans = {}
    if city == "beijing":
        link = 'https://www.wunderground.com/history/airport/ZBAA/%d/%02d/%02d/DailyHistory.html?req_city=Beijing&req_statename=China' % (
            year, month, day)
    else:
        link = 'https://www.wunderground.com/history/airport/EGMC/%d/%02d/%02d/DailyHistory.html?req_city=London&req_statename=England' % (
            year, month, day)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6',
               'referer': 'link'}
    beautiful = requests.get(link, headers=headers).text.decode('utf-8', 'ignore')
    soup = bs4.BeautifulSoup(unicodedata.normalize("NFKD", beautiful), 'html.parser')
    table = soup.find(id="obsTable")
    # print table
    tr_list = table.find("tbody").find_all("tr")
    # print tr_list[0]
    tmp = []
    for tr in tr_list:
        # print tr
        td_list = tr.find_all("td")
        # print td_list
        time_ = td_list[0].get_text(strip=True).strip()  # 时间
        temp_ = td_list[1].get_text(strip=True).strip("\n").strip("\r").strip().replace("\t", "")  # 气温
        windchill_ = td_list[2].get_text(strip=True).replace("\n", "").replace("\r", "").replace("\t",
                                                                                                 "").strip()  # 风冷温
        dew_point_ = td_list[3].get_text(strip=True).replace("\n", "").replace("\r", "").replace("\t", "").strip()  # 露点
        humidity_ = td_list[4].get_text(strip=True).replace("\n", "").replace("\r", "").replace("\t", "").strip()  # 湿度
        pressure_ = td_list[5].get_text(strip=True).replace("\n", "").replace("\r", "").replace("\t", "").strip()  # 气压
        visiability_ = td_list[6].get_text(strip=True).replace("\n", "").replace("\r", "").replace("\t",
                                                                                                   "").strip()  # 能见度
        wind_dir_ = td_list[7].get_text(strip=True).replace("\n", "").replace("\r", "").replace("\t", "").strip()  # 风向
        wind_speed_ = td_list[8].get_text(strip=True).replace("\n", "").replace("\r", "").replace("\t",
                                                                                                  "").strip()  # 风速度
        gust_speed_ = td_list[9].get_text(strip=True).replace("\n", "").replace("\r", "").replace("\t",
                                                                                                  "").strip()  # 瞬间风速
        precip_ = td_list[10].get_text(strip=True).replace("\n", "").replace("\r", "").replace("\t",
                                                                                               "").strip()  # precip
        events_ = td_list[11].get_text(strip=True).replace("\n", "").replace("\r", "").replace("\t", "").strip()  # 活动
        conditions_ = td_list[12].get_text(strip=True).replace("\n", "").replace("\r", "").replace("\t",
                                                                                                   "").strip()  # 状况
        tmp_1 = time_ + ',' + temp_ + ',' + windchill_ + ',' + dew_point_ + ',' + humidity_ + ',' + pressure_ + ',' + visiability_ + ',' + wind_dir_ + ',' + wind_speed_ + ',' + gust_speed_ + ',' + precip_ + ',' + events_ + ',' + conditions_
        tmp_1 = tmp_1.encode('unicode-escape').replace(r'\xb0C', r'')
        tmp_1 = tmp_1.replace("m/s", "").replace("%", "").replace("hPa", "").replace("km/h()", " ").replace(
            "kilometers", "").replace("km/h", "").replace("mm", "")
        tmp_1 = tmp_1.strip()
        tmp.append(tmp_1.split(","))
    ans["hours_data"] = tmp

    tmp = ""
    table = soup.find(id="historyTable")
    tr_list = table.find("tbody").find_all("tr")
    temp_avg_ = tr_list[1].find_all("td")[1].get_text(strip=True).replace("\n", "").replace("\r", "").replace("\t",
                                                                                                              "").strip()
    temp_max_ = tr_list[2].find_all("td")[1].get_text(strip=True).replace("\n", "").replace("\r", "").replace("\t",
                                                                                                              "").strip()
    temp_min_ = tr_list[3].find_all("td")[1].get_text(strip=True).replace("\n", "").replace("\r", "").replace("\t",
                                                                                                              "").strip()
    degree_day_ = tr_list[5].find_all("td")[1].get_text(strip=True).replace("\n", "").replace("\r", "").replace("\t",
                                                                                                                "").strip()
    dew_point_ = tr_list[7].find_all("td")[1].get_text(strip=True).replace("\n", "").replace("\r", "").replace("\t",
                                                                                                               "").strip()
    humidity_avg_ = tr_list[8].find_all("td")[1].get_text(strip=True).replace("\n", "").replace("\r", "").replace("\t",
                                                                                                                  "").strip()
    humidity_max_ = tr_list[9].find_all("td")[1].get_text(strip=True).replace("\n", "").replace("\r", "").replace("\t",
                                                                                                                  "").strip()
    humidity_min_ = tr_list[10].find_all("td")[1].get_text(strip=True).replace("\n", "").replace("\r", "").replace("\t",
                                                                                                                   "").strip()
    rain_value_ = tr_list[12].find_all("td")[1].get_text(strip=True).replace("\n", "").replace("\r", "").replace("\t",
                                                                                                                 "").strip()
    pressure_ = tr_list[14].find_all("td")[1].get_text(strip=True).replace("\n", "").replace("\r", "").replace("\t",
                                                                                                               "").strip()
    speed_avg_ = tr_list[16].find_all("td")[1].get_text(strip=True).replace("\n", "").replace("\r", "").replace("\t",
                                                                                                                "").strip()
    speed_max_ = tr_list[17].find_all("td")[1].get_text(strip=True).replace("\n", "").replace("\r", "").replace("\t",
                                                                                                                "").strip()
    visiability_ = tr_list[19].find_all("td")[1].get_text(strip=True).replace("\n", "").replace("\r", "").replace("\t",
                                                                                                                  "").strip()
    events_ = tr_list[20].find_all("td")[1].get_text(strip=True).replace("\n", "").replace("\r", "").replace("\t",
                                                                                                             "").strip()
    tmp_1 = temp_avg_ + "," + temp_max_ + "," + temp_min_ + "," + degree_day_ + "," + dew_point_ + "," + humidity_avg_ + "," + humidity_max_ + "," + humidity_min_ + "," + rain_value_ + "," + pressure_ + "," + speed_avg_ + "," + speed_max_ + "," + visiability_ + "," + events_
    tmp_1 = tmp_1.encode('unicode-escape').replace(r'\xb0C', r'')
    tmp_1 = tmp_1.replace("m/s", "").replace("%", "").replace("hPa", "").replace("km/h()", " ").replace("kilometers",
                                                                                                        "").replace(
        "km/h", "").replace("mm", "")
    tmp_1 = tmp_1.strip()
    ans["day_data"] = tmp_1.split(",")
    return ans


# https://www.wunderground.com/personal-weather-station/dashboard?ID=ILONDON636#history/s20180405/e20180405/mdaily
# https://www.wunderground.com/hourly/cn/baizhifang/date/2018-04-08/I11BAIZH2?cm_ven=localwx_hour
'''
# 根据站点的id获取天气,包含历史,当天和预测

data = {
    "history": {
        "days": {
            [{
                "observations": [{}, {}],  # 一天每小时的天气情况
                "summary": {}  # 一天整体的天气
            }]
        },
        "end_date": {},
        "start_date": {},
    },
    "response": {
        "date": {}  # 查询当天的天气情况
    }
}

'''


def weather_station(year=2018, month=4, day=6, station_id="I11BAIZH2"):
    time_str = "%d%02d%02d" % (year, month, day)
    time_find = parser.parse(time_str)
    time_find = time_find.strftime('%Y-%m-%d')

    time_now = time.time()
    time_now = time.localtime(time_now)
    time_now = time.strftime('%Y-%m-%d', time_now)
    if time_now > time_find:
        link = 'https://www.wunderground.com/personal-weather-station/dashboard?ID=' + station_id + '#history/s' + time_str + '/e' + time_str + '/mdaily'
    else:
        link = 'https://www.wunderground.com/hourly/date/' + str(time_find) + '/' + station_id + '?cm_ven=localwx_hour'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6',
               'referer': 'link'}
    beautiful = requests.get(link, headers=headers).text.decode('utf-8', 'ignore')
    soup = bs4.BeautifulSoup(unicodedata.normalize("NFKD", beautiful), 'html.parser')
    ans = {}
    print time_now, time_find
    if time_now > time_find:
        link1 = 'https://api-ak.wunderground.com/api/606f3f6977348613/history_' + time_str + 'null/units:metric/v:2.0/q/pws:' + station_id + '.json?'
        print link1
        respones1 = requests.get(link1)
        text = respones1.text
        # print text
        data = json.loads(text)
        # print data
        ans["history"] = data["history"]["days"][0]
        ans["today"] = data["response"]["date"]
    else:
        lat = float(soup.find("span", {"class": "subheading"}).find_all("strong")[1].get_text(strip=True).strip())
        lng = float(soup.find("span", {"class": "subheading"}).find_all("strong")[2].get_text(strip=True).strip())
        ans = weather_lat_lng(lat, lng)
        # table = soup.find(id = "hourly-forecast-table")
        # tr_list = table.find("tbody").find_all("tr")
        # # print tr_list
        # tmp = []
        # for tr in tr_list:
        # 	td_list = tr.find_all('td')
        # 	time_ = td_list[0].get_text(strip=True).replace("\n","").replace("\r","").replace("\t","").strip()
        # 	conditions_ = td_list[1].find("span",{"class":"show-for-medium conditions"}).get_text(strip=True).replace("\n","").replace("\r","").replace("\t","").strip()
        # 	temp_ = td_list[2].get_text(strip=True).replace("\n","").replace("\r","").replace("\t","").strip()
        # 	feels_like_ = td_list[3].get_text(strip=True).replace("\n","").replace("\r","").replace("\t","").strip()
        # 	precip_ = td_list[4].get_text(strip=True).replace("\n","").replace("\r","").replace("\t","").strip()
        # 	amount_ = td_list[5].get_text(strip=True).replace(" in","").replace("\n","").replace("\r","").replace("\t","").strip()
        # 	cloud_cover_ = td_list[6].get_text(strip=True).replace("\n","").replace("\r","").replace("\t","").strip()
        # 	dew_point_ = td_list[7].get_text(strip=True).replace("\n","").replace("\r","").replace("\t","").strip()
        # 	humidity_ = td_list[8].get_text(strip=True).replace("\n","").replace("\r","").replace("\t","").strip()
        # 	wind_ = td_list[9].get_text(strip=True).replace("\n","").replace("\r","").replace("\t","").strip()
        # 	wind_speed_ = wind_.split(" ")[0]
        # 	wind_dir_ = wind_.split(" ")[2]
        # 	pressure_ = td_list[10].find("span",{"class":"wu-value wu-value-to"}).get_text(strip=True).replace("\n","").replace("\r","").replace("\t","").strip()
        # 	tmp_1 = time_+","+conditions_+","+temp_+","+feels_like_+","+precip_+","+amount_+","+cloud_cover_+","+dew_point_+","+humidity_+","+wind_speed_+","+wind_dir_+","+pressure_
        # 	tmp_1 = tmp_1.encode('unicode-escape').replace(r'\xb0F',r'')
        # 	tmp_1 = tmp_1.replace("m/s","").replace("%","").replace("hPa","").replace("km/h()"," ").replace("kilometers","").replace("km/h","").replace("mm","")
        # 	tmp_1 = tmp_1.strip()
        # 	tmp.append(tmp_1.split(","))
        # ans["hours_data"] = tmp
        pass
    return ans


# https://stationdata.wunderground.com/cgi-bin/stationdata?v=2.0&type=ICAO%2CPWS&units=english&format=json&maxage=1800&maxstations=35&minstations=10&centerLat=39.873821&centerLon=116.339943&height=400&width=400&iconsize=2
# https://api.weather.com/v1/geocode/40.039/116.395/forecast/hourly/240hour.json?apiKey=6532d6454b8aa370768e63d6ba5a832e&units=e
'''
根据经纬度获取附近站点的天气情况
根据经纬度获取近10天24小时的天气数据
'''


def weather_lat_lng(lat, lng):
    link1 = 'https://stationdata.wunderground.com/cgi-bin/stationdata?v=2.0&type=ICAO%2CPWS&units=english&format=json&maxage=1800&maxstations=35&minstations=10&centerLat=' + str(
        lat) + '&centerLon=' + str(lng) + '&height=400&width=400&iconsize=2'
    link2 = 'https://api.weather.com/v1/geocode/' + str(lat) + '/' + str(
        lng) + '/forecast/hourly/240hour.json?apiKey=6532d6454b8aa370768e63d6ba5a832e&units=e'
    link3 = 'https://api.weather.com/v1/geocode/' + str(lat) + '/' + str(
        lng) + '/forecast/daily/10day.json?apiKey=6532d6454b8aa370768e63d6ba5a832e&units=e'
    link4 = 'https://api-ak.wunderground.com/api/d8585d80376a429e/labels/hourly10day/lang:EN/units:english/bestfct:1/v:2.0/q/' + str(
        lat) + ',' + str(lng) + '.json?ttl=120'

    # try:
    #     respones1 = requests.get(link1)
    #     stations = json.loads(respones1.text)["stations"]
    # except:
    #     stations = []
    respones2 = requests.get(link2)
    forecasts_hours = json.loads(respones2.text)["forecasts"]
    # respones3 = requests.get(link3)
    # forecasts_daily = json.loads(respones3.text)["forecasts"]
    respones4 = requests.get(link4)
    text = respones4.text
    print link4
    print json.loads(text)
    try:
        history_hours = json.loads(text)["history"]['days'][0]['hours']
    except:
        history_hours = []
    ans = {}
    # ans["stations"] = stations
    ans["forecasts_hours"] = forecasts_hours
    # ans['forecasts_daily'] = forecasts_daily
    ans['history_hours'] = history_hours
    return ans


def find_station_information():
    # fr_to = open("stataion_beijing.txt","w")
    fr_to = open("stataion_london.txt", "w")
    seed_lat = 51.51
    seed_lng = 0.13
    station_informain = {}
    import Queue
    q = Queue.Queue()
    q.put([seed_lat, seed_lng])
    while (True):
        tmp = q.get()
        ans = weather_lat_lng(tmp[0], tmp[1])
        stations = ans["stations"]
        for i in range(len(stations)):
            id = stations[i]['id']
            latitude = stations[i]['latitude']
            longitude = stations[i]['longitude']
            adm1 = stations[i]['adm1']
            if station_informain.has_key(id) == False:
                station_informain[id] = {}
                q.put([latitude, longitude])
            station_informain[id]["latitude"] = latitude
            station_informain[id]["longitude"] = longitude
            station_informain[id]["adm1"] = adm1
        if q.empty():
            break
        if len(station_informain.keys()) > 100:
            break
    print station_informain
    print len(station_informain.keys())
    for key in station_informain.keys():
        fr_to.write(str(key) + "," + str(station_informain[key]['latitude']) + "," + str(
            station_informain[key]['longitude']) + "," + station_informain[key]['adm1'] + "\n")
    fr_to.close()


# https://api.waqi.info/mapq/bounds/?bounds=39.39799959542146,115.04608154296876,40.073868105094846,117.59765625000001&inc=placeholders&k=_2Y2EzUR9IAR8fDStDSElWXmldWkc+EzMZFngzZA==&_=1523093789592
# http://aqicn.org/city/beijing
# https://api.waqi.info/api/feed/@1451/obs.en.json
# http://aqicn.org/city/london/
# https://api.waqi.info/api/feed/@5724/obs.en.json

def aqi_idx(idx=5724):
    link = 'https://api.waqi.info/api/feed/@' + str(idx) + '/obs.en.json'
    respones = requests.get(link)
    msg = json.loads(respones.text)["rxs"]["obs"][0]["msg"]
    print msg


def main():
    year = 2018
    month = 4
    day = 8
    city = "London"
    # ans = city_weather(year=year, month=month, day=day, city=city)
    # print ans
    weather_station(year=year, month=month, day=day)
    lat = 40.039
    lng = 116.395
    # weather_lat_lng(lat = lat, lng = lng)
    # find_station_information()
    # aqi_idx()


if __name__ == '__main__':
    main()
