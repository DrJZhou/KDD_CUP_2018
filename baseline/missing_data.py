from __future__ import division
from __future__ import print_function
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute, MICE
from nose.tools import eq_
import cPickle as pickle
import xlrd

base_path_1 = "../dataset/"
base_path_2 = "../dataset/tmp/"
base_path_3 = "../output/"


def load_station():
    filename = base_path_1 + "Beijing_AirQuality_Stations_cn.xlsx"
    data = xlrd.open_workbook(filename)
    table = data.sheet_by_name(u'Sheet2')
    nrows = table.nrows
    bj_stations = {}
    for i in range(1, nrows):
        row = table.row_values(i)
        # print row
        bj_stations[row[0]] = {}
        bj_stations[row[0]]['lng'] = row[1]
        bj_stations[row[0]]['lat'] = row[2]
        bj_stations[row[0]]['type_id'] = int(row[-1])
        bj_stations[row[0]]['station_num_id'] = i

    filename = base_path_1 + "London_AirQuality_Stations.csv"
    fr = open(filename)
    ld_stations = {}
    flag = 0
    i = 0
    for line in fr.readlines():
        if flag == 0:
            flag = 1
            continue
        row = line.strip().split(",")
        ld_stations[row[0]] = {}
        if row[2] == "TRUE":
            ld_stations[row[0]]['predict'] = True
        else:
            ld_stations[row[0]]['predict'] = False
        ld_stations[row[0]]['lng'] = float(row[5])
        ld_stations[row[0]]['lat'] = float(row[4])
        ld_stations[row[0]]['type_id'] = int(row[-1])
        ld_stations[row[0]]['station_num_id'] = i
        i += 1
    stations = {}
    stations["bj"] = bj_stations
    stations["ld"] = ld_stations
    return stations


def RFImputer(Ximp):
    mask = np.isnan(Ximp)
    missing_rows, missing_cols = np.where(mask)

    # MissForest Algorithm
    # 1. Make initial guess for missing values
    col_means = np.nanmean(Ximp, axis=0)
    Ximp[(missing_rows, missing_cols)] = np.take(col_means, missing_cols)

    # 2. k <- vector of sorted indices of columns in X
    col_missing_count = mask.sum(axis=0)
    k = np.argsort(col_missing_count)

    # 3. While not gamma_new < gamma_old and iter < max_iter  do:
    iter = 0
    max_iter = 100
    gamma_new = 0
    gamma_old = np.inf
    col_index = np.arange(Ximp.shape[1])
    model_rf = RandomForestRegressor(random_state=0, n_estimators=1000)
    # TODO: Update while condition for categorical vars
    while gamma_new < gamma_old and iter < max_iter:
        # added
        # 4. store previously imputed matrix
        Ximp_old = np.copy(Ximp)
        if iter != 0:
            gamma_old = gamma_new
        # 5. loop
        for s in k:
            s_prime = np.delete(col_index, s)
            obs_rows = np.where(~mask[:, s])[0]
            mis_rows = np.where(mask[:, s])[0]
            yobs = Ximp[obs_rows, s]
            xobs = Ximp[np.ix_(obs_rows, s_prime)]
            xmis = Ximp[np.ix_(mis_rows, s_prime)]
            # 6. Fit a random forest
            model_rf.fit(X=xobs, y=yobs)
            # 7. predict ymis(s) using xmis(x)
            ymis = model_rf.predict(xmis)
            Ximp[mis_rows, s] = ymis
            # 8. update imputed matrix using predicted matrix ymis(s)
        # 9. Update gamma
        gamma_new = np.sum((Ximp_old - Ximp) ** 2) / np.sum(
            (Ximp) ** 2)
        print("Iteration:", iter)
        iter += 1
    return Ximp_old


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


def impute(city, methods="KNN"):
    filename = base_path_2 + city + "_airquality_processing.csv"
    if city == 'bj':
        attr_need = ["station_id_num", "PM25_Concentration", "PM10_Concentration", "O3_Concentration", "time_week",
                     "time_month", "time_day", "time_hour", "CO_Concentration", "NO2_Concentration",
                     "SO2_Concentration"]
    else:
        attr_need = ["station_id_num", "PM25_Concentration", "PM10_Concentration", "time_week",
                     "time_month", "time_day", "time_hour", "NO2_Concentration"]
    df = pd.read_csv(filename, sep=',')
    df['time'] = pd.to_datetime(df['time'])
    df.index = df['time']
    df[df < 0] = np.nan
    station_groups = df.groupby(['station_id'])
    stations = load_station()
    city_station = stations[city]
    stations_group = {}
    for station, group in station_groups:
        df1 = group
        df1['station_id_num'] = df1.apply(
            lambda row: float(city_station[str(row.station_id)]['station_num_id']), axis=1)
        XY_incomplete = df1[attr_need].values
        # print(XY_incomplete)
        if methods == "KNN":
            XY_completed = KNN(k=5).complete(XY_incomplete)
        # print(XY_completed)
        if methods == "MICE":
            # print(XY_incomplete)
            try:
                XY_completed = MICE(n_imputations=100).complete(XY_incomplete)
            except:
                continue
        # print(XY_completed)
        group.loc[:, attr_need] = XY_completed
        stations_group[station] = group
    import cPickle as pickle
    f1 = file(base_path_3 + city + '_data_history_'+methods+'.pkl', 'wb')
    pickle.dump(stations_group, f1, True)


# def knn(XY_incomplete, k=5):
#     XY_completed = KNN(k).complete(XY_incomplete)
#     return XY_completed


def get_loss_rate():
    filename = base_path_2 + "rate.pkl"
    f1 = file(filename, 'wb')
    loss_rate = {}
    loss_rate['bj'] = loss_data_day(city='bj')
    loss_rate['ld'] = loss_data_day(city='ld')
    pickle.dump(loss_rate, f1, True)


if __name__ == '__main__':
    # main(city='bj')
    # loss_data_day('bj')
    get_loss_rate()
    # impute(city='bj', methods="KNN")
    # impute(city='ld', methods="KNN")
    # impute(city='bj', methods="MICE")
    # impute(city='ld', methods="MICE")
