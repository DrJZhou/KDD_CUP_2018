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

# from rule import *
reload(sys)
sys.setdefaultencoding('utf-8')
base_path_1 = "../dataset/"
base_path_2 = "../dataset/tmp/"
base_path_3 = "../output/"


def ensemble(file1, file2, file3, a):
    df1 = pd.read_csv(file1, sep=',')
    df1.index = df1['test_id']
    df1 = df1.drop(["test_id"], axis=1)
    df2 = pd.read_csv(file2, sep=',')
    df2.index = df2['test_id']
    df2 = df2.drop(["test_id"], axis=1)
    # df2 = df2.sort_index()
    df3 = df1 * a + df2 * (1 - a)
    df3.to_csv(file3, index=True, sep=',')


def ensemble_three_file(file1, file2, file3, file4, a, b, c):
    df1 = pd.read_csv(file1, sep=',')
    df1.index = df1['test_id']
    df1 = df1.drop(["test_id"], axis=1)
    df2 = pd.read_csv(file2, sep=',')
    df2.index = df2['test_id']
    df2 = df2.drop(["test_id"], axis=1)
    df3 = pd.read_csv(file3, sep=',')
    df3.index = df3['test_id']
    df3 = df3.drop(["test_id"], axis=1)
    df4 = df1 * a + df2 * b + df3 * c
    df4.to_csv(file4, index=True, sep=',')


from model import test

'''
type_="0301-0531_0801-0410" , feature_first=True
type_="0301-0531_0801-0410" , feature_first=False
type_="2017_0101-2018_0410_less" , feature_first=False
ans_file = base_path_3 + "test/" + end_day + "-xgboost_weather" + type + "_" + str(feature_first) + ".csv"
'''


def cal_ensemble_best_xgboost():
    score = np.zeros(6)
    num = 0.0
    for j in range(11, 28):
        end_day = "2018-04-" + str(j)
        file1 = base_path_3 + "test/" + end_day + "-xgboost_weather" + "0301-0531_0801-0410" + "_" + str(
            True) + ".csv"
        file2 = base_path_3 + "test/" + end_day + "-xgboost_weather" + "2017_0101-2018_0410_less" + "_" + str(
            False) + ".csv"
        file3 = base_path_3 + "test/" + end_day + "-xgboost_weather" + "0301-0531_0801-0410" + "_" + str(
            False) + ".csv"
        file6 = base_path_3 + "test/" + end_day + "-ext_with_weather_" + "0301-0531_0801-0410" + "_" + str(
            False) + ".csv"
        file4 = base_path_3 + end_day + "xgboost_mean_ensemble.csv"
        file5 = base_path_3 + end_day + "xgboost_median_ensemble.csv"
        ensemble_median(file1, file2, file3, file6, file4, file5)
        # ensemble_mean_3(file1, file2, file3, file4, file5)

        score1, _, _ = test(end_day, file1)
        score2, _, _ = test(end_day, file2)
        score3, _, _ = test(end_day, file3)
        score6, _, _ = test(end_day, file6)
        score4, _, _ = test(end_day, file4)
        score5, _, _ = test(end_day, file5)
        score_now = np.array([score1, score2, score3, score6, score4, score5])
        score += score_now
        print "score: ", score_now
        num += 1.0
    avg_score = score / num
    print "avg_score: ", avg_score

    # a = 0.3
    # total_score = 0.0
    # num = 0.0
    # for j in range(11, 28):
    #     end_day = "2018-04-" + str(j)
    #     file1 = base_path_3 + "test/" + end_day + "-xgboost_weather" + "0301-0531_0801-0410" + "_" + str(
    #         True) + ".csv"
    #     file2 = base_path_3 + "test/" + end_day + "-xgboost_weather" + "2017_0101-2018_0410_less" + "_" + str(
    #         False) + ".csv"
    #     file3 = base_path_3 + "test/" + end_day + "-xgboost_weather" + "0301-0531_0801-0410" + "_" + str(
    #         False) + ".csv"
    #     file4 = base_path_3 + "xgboost_ensemble.csv"
    #     ensemble(file2, file3, file4, a)
    #     score, _, _ = test(end_day, file4)
    #     print "score: ", score
    #     total_score += score
    #     num += 1.0
    # avg_score = total_score / num
    # print "avg_score: ", avg_score

    # best_params = [0.0, 0.0, 0.0]
    # best_score = 2.0
    # for i in range(11):
    #     a = i * 1.0 / 10.0
    #     for k in range(11):
    #         b = k * 1.0 / 10.0
    #         if a + b > 1.0:
    #             continue
    #         c = 1 - a - b
    #         total_score = 0.0
    #         num = 0.0
    #         for j in range(11, 28):
    #             end_day = "2018-04-" + str(j)
    #             file1 = base_path_3 + "test/" + end_day + "-xgboost_weather" + "0301-0531_0801-0410" + "_" + str(
    #                 True) + ".csv"
    #             file2 = base_path_3 + "test/" + end_day + "-xgboost_weather" + "2017_0101-2018_0410_less" + "_" + str(
    #                 False) + ".csv"
    #             file3 = base_path_3 + "test/" + end_day + "-xgboost_weather" + "0301-0531_0801-0410" + "_" + str(
    #                 False) + ".csv"
    #             file4 = base_path_3 + "xgboost_ensemble.csv"
    #             ensemble_three_file(file1, file2, file3, file4=file4, a=a, b=b, c=c)
    #             score, _, _ = test(end_day, file4)
    #             print "score: ", score
    #             total_score += score
    #             num += 1.0
    #         avg_score = total_score / num
    #         print "avg_score: ", avg_score
    #         if avg_score < best_score:
    #             best_params = [a, b, c]
    #             best_score = avg_score
    # print best_params


'''
get_test(type="0301-0531_0801-0410", feature_first=False)
get_test(type="0301-0531_0801-0410", feature_first=True)
get_test(type="2017_0101-2018_0410", feature_first=True)
get_test(type="2017_0101-2018_0410_less", feature_first=False)
ans_file = base_path_3 + "test/" + end_day + "-ext_with_weather_" + type + "_" + str(feature_first) + ".csv"
'''


def cal_ensemble_best_ext_with_weather():
    # for i in range(11):
    a = 1.0
    b = 0.0
    c = 0.0
    total_score = 0.0
    num = 0.0
    for j in range(11, 28):
        end_day = "2018-04-" + str(j)
        file1 = base_path_3 + "test/" + end_day + "-ext_with_weather_" + "0301-0531_0801-0410" + "_" + str(
            False) + ".csv"
        file2 = base_path_3 + "test/" + end_day + "-ext_with_weather_" + "0301-0531_0801-0410" + "_" + str(
            True) + ".csv"
        file3 = base_path_3 + "test/" + end_day + "-ext_with_weather_" + "2017_0101-2018_0410_less" + "_" + str(
            False) + ".csv"
        file4 = base_path_3 + "ext_with_weather_ensemble.csv"
        ensemble_three_file(file1, file2, file3, file4=file4, a=a, b=b, c=c)
        score, _, _ = test(end_day, file4)
        print "score: ", score
        total_score += score
        num += 1.0
    avg_score = total_score / num
    print "avg_score: ", avg_score

    # best_params = [0.0, 0.0, 0.0]
    # best_score = 2.0
    # for i in range(11):
    #     a = i * 1.0 / 10.0
    #     for k in range(11):
    #         b = k*1.0/10.0
    #         if a + b > 1.0:
    #             continue
    #         c = 1.0 - a -b
    #         total_score = 0.0
    #         num = 0.0
    #         for j in range(11, 28):
    #             end_day = "2018-04-" + str(j)
    #             file1 = base_path_3 + "test/" + end_day + "-ext_with_weather_" + "0301-0531_0801-0410" + "_" + str(False) + ".csv"
    #             file2 = base_path_3 + "test/" + end_day + "-ext_with_weather_" + "0301-0531_0801-0410" + "_" + str(True) + ".csv"
    #             file3 = base_path_3 + "test/" + end_day + "-ext_with_weather_" + "2017_0101-2018_0410_less" + "_" + str(False) + ".csv"
    #             file4 = base_path_3 + "ext_with_weather_ensemble.csv"
    #             ensemble_three_file(file1, file2, file3, file4=file4, a=a, b=b, c=c)
    #             score, _, _ = test(end_day, file4)
    #             print "score: ", score
    #             total_score += score
    #             num += 1.0
    #         avg_score = total_score / num
    #         print "avg_score: ", avg_score
    #         if avg_score < best_score:
    #             best_params = [a, b, b]
    #             best_score = avg_score
    # print best_params


'''
get_test(type="0301-0531_0801-0410")
get_test(type="2017_0101-2018_0410_less")
get_test(type="2017_0101-2018_0410_test")
ans_file = base_path_3 + "test/" + end_day + "_ext_with_weather_three_metric_" + type + ".csv"
'''


def cal_ensemble_best_ext_with_weather_three_metric():
    # for i in range(11):
    a = 1.0
    b = 0.0
    c = 0.0
    total_score = 0.0
    num = 0.0
    for j in range(11, 28):
        end_day = "2018-04-" + str(j)
        file1 = base_path_3 + "test/" + end_day + "_ext_with_weather_three_metric_" + "0301-0531_0801-0410" + ".csv"
        file2 = base_path_3 + "test/" + end_day + "_ext_with_weather_three_metric_" + "2017_0101-2018_0410_less" + ".csv"
        file3 = base_path_3 + "test/" + end_day + "_ext_with_weather_three_metric_" + "2017_0101-2018_0410_test" + ".csv"
        file4 = base_path_3 + "ext_with_weather_three_metric_ensemble.csv"
        ensemble_three_file(file1, file2, file3, file4=file4, a=a, b=b, c=c)
        score, _, _ = test(end_day, file4)
        print "score: ", score
        total_score += score
        num += 1.0
    avg_score = total_score / num
    print "avg_score: ", avg_score

    # best_params = [0.0, 0.0, 0.0]
    # best_score = 2.0
    # for i in range(11):
    #     a = i * 1.0 / 10.0
    #     for k in range(11):
    #         b = k*1.0/10.0
    #         if a + b > 1.0:
    #             continue
    #         c = 1.0 - a -b
    #         total_score = 0.0
    #         num = 0.0
    #         for j in range(11, 28):
    #             end_day = "2018-04-" + str(j)
    #             file1 = base_path_3 + "test/" + end_day + "_ext_with_weather_three_metric_" + "0301-0531_0801-0410" + ".csv"
    #             file2 = base_path_3 + "test/" + end_day + "_ext_with_weather_three_metric_" + "2017_0101-2018_0410_less" + ".csv"
    #             file3 = base_path_3 + "test/" + end_day + "_ext_with_weather_three_metric_" + "2017_0101-2018_0410_test" + ".csv"
    #             file4 = base_path_3 + "ext_with_weather_three_metric_ensemble.csv"
    #             ensemble_three_file(file1, file2, file3, file4=file4, a=a, b=b, c=c)
    #             score, _, _ = test(end_day, file4)
    #             print "score: ", score
    #             total_score += score
    #             num += 1.0
    #         avg_score = total_score / num
    #         print "avg_score: ", avg_score
    #         if avg_score < best_score:
    #             best_params = [a, b, b]
    #             best_score = avg_score
    # print best_params


def cal_ensemble_all():
    a = 0.6
    total_score = 0.0
    total_score1 = 0.0
    total_score2 = 0.0
    num = 0.0
    for j in range(11, 28):
        end_day = "2018-04-" + str(j)
        file1 = base_path_3 + "test/" + end_day + "-ext_with_weather_" + "0301-0531_0801-0410" + "_" + str(
            False) + ".csv"
        file2 = base_path_3 + "test/" + end_day + "-ext_with_weather_" + "0301-0531_0801-0410" + "_" + str(
            True) + ".csv"
        file3 = base_path_3 + "test/" + end_day + "-ext_with_weather_" + "2017_0101-2018_0410_less" + "_" + str(
            False) + ".csv"
        file4 = base_path_3 + "ext_with_weather_ensemble.csv"
        # ensemble_three_file(file1, file2, file3, file4=file4, a=1.0, b=0.0, c=0.0)

        file5 = base_path_3 + "test/" + end_day + "-xgboost_weather" + "2017_0101-2018_0410_less" + "_" + str(
            False) + ".csv"
        file6 = base_path_3 + "test/" + end_day + "-xgboost_weather" + "0301-0531_0801-0410" + "_" + str(
            False) + ".csv"
        file7 = base_path_3 + "xgboost_ensemble.csv"
        ensemble(file5, file6, file7, a=0.3)

        file8 = base_path_3 + "test/" + end_day + "_ext_with_weather_three_metric_" + "0301-0531_0801-0410" + ".csv"
        file9 = base_path_3 + "test/" + end_day + "_ext_with_weather_three_metric_" + "2017_0101-2018_0410_less" + ".csv"
        file10 = base_path_3 + "test/" + end_day + "_ext_with_weather_three_metric_" + "2017_0101-2018_0410_test" + ".csv"
        file11 = base_path_3 + "ext_with_weather_three_metric_ensemble.csv"
        # ensemble_three_file(file8, file9, file10, file4=file11, a=1.0, b=0.0, c=0.0)

        file12 = base_path_3 + "ensemble_all_1.csv"
        ensemble(file1, file7, file12, a=0.6)
        score, _, _ = test(end_day, file12)
        print "score_1: ", score
        total_score1 += score

        print "after rule:", score
        file5 = base_path_3 + "test/" + end_day + "-xgboost_weather" + "0301-0531_0801-0410" + "_" + str(
            True) + ".csv"
        file6 = base_path_3 + "test/" + end_day + "-xgboost_weather" + "2017_0101-2018_0410_less" + "_" + str(
            False) + ".csv"
        file7 = base_path_3 + "xgboost_ensemble.csv"
        ensemble(file5, file6, file7, a=0.2)

        file13 = base_path_3 + "ensemble_all_2.csv"
        ensemble(file1, file7, file13, a=0.7)
        score, _, _ = test(end_day, file13)
        total_score2 += score
        print "score_2: ", score

        file14 = base_path_3 + end_day + "_ensemble_zhoujie.csv"
        ensemble(file12, file13, file14, a=0.6)
        score, _, _ = test(end_day, file14)
        total_score += score
        print "score: ", score
        num += 1.0
    avg_score1 = total_score1 / num
    avg_score2 = total_score2 / num
    avg_score = total_score / num
    print "avg_score: ", avg_score1, avg_score2, avg_score

    # best_params = [0.0]
    # best_score = 2.0
    # for i in range(11):
    #     a = i * 1.0 / 10.0
    #     total_score = 0.0
    #     num = 0.0
    #     for j in range(11, 28):
    #         end_day = "2018-04-" + str(j)
    #         file1 = base_path_3 + "test/" + end_day + "-ext_with_weather_" + "0301-0531_0801-0410" + "_" + str(
    #             False) + ".csv"
    #         file2 = base_path_3 + "test/" + end_day + "-ext_with_weather_" + "0301-0531_0801-0410" + "_" + str(
    #             True) + ".csv"
    #         file3 = base_path_3 + "test/" + end_day + "-ext_with_weather_" + "2017_0101-2018_0410_less" + "_" + str(
    #             False) + ".csv"
    #         file4 = base_path_3 + "ext_with_weather_ensemble.csv"
    #         # ensemble_three_file(file1, file2, file3, file4=file4, a=1.0, b=0.0, c=0.0)
    #
    #         # file5 = base_path_3 + "test/" + end_day + "-xgboost_weather" + "0301-0531_0801-0410" + "_" + str(
    #         #     True) + ".csv"
    #         file5 = base_path_3 + "test/" + end_day + "-xgboost_weather" + "2017_0101-2018_0410_less" + "_" + str(
    #             False) + ".csv"
    #         file6 = base_path_3 + "test/" + end_day + "-xgboost_weather" + "0301-0531_0801-0410" + "_" + str(
    #             False) + ".csv"
    #         file7 = base_path_3 + "xgboost_ensemble.csv"
    #         ensemble(file5, file6, file7, a=0.3)
    #
    #         file8 = base_path_3 + "test/" + end_day + "_ext_with_weather_three_metric_" + "0301-0531_0801-0410" + ".csv"
    #         file9 = base_path_3 + "test/" + end_day + "_ext_with_weather_three_metric_" + "2017_0101-2018_0410_less" + ".csv"
    #         file10 = base_path_3 + "test/" + end_day + "_ext_with_weather_three_metric_" + "2017_0101-2018_0410_test" + ".csv"
    #         file11 = base_path_3 + "ext_with_weather_three_metric_ensemble.csv"
    #         # ensemble_three_file(file8, file9, file10, file4=file11, a=1.0, b=0.0, c=0.0)
    #
    #         file12 = base_path_3 + "ensemble_all.csv"
    #         # ensemble_three_file(file4, file4, file11, file4=file12, a=a, b=b, c=c)
    #         # ensemble_three_file(file1, file7, file8, file4=file12, a=a, b=b, c=c)
    #         ensemble(file1, file7, file12, a=a)
    #         score, _, _ = test(end_day, file12)
    #         print "score: ", score
    #         total_score += score
    #         num += 1.0
    #     avg_score = total_score / num
    #     print "avg_score: ", avg_score
    #     if avg_score < best_score:
    #         best_params = [a]
    #         best_score = avg_score
    # print best_params


def get_ans_1(end_day, caiyun=False):
    if caiyun == False:
        file1 = base_path_3 + "" + end_day + "-ext_with_weather_" + "0301-0531_0801-0410" + "_" + str(
            False) + ".csv"
        file5 = base_path_3 + "" + end_day + "-xgboost_weather" + "0301-0531_0801-0410" + "_" + str(
            True) + ".csv"
        file6 = base_path_3 + "" + end_day + "-xgboost_weather" + "2017_0101-2018_0410_less" + "_" + str(
            False) + ".csv"
        file7 = base_path_3 + "xgboost_ensemble.csv"
    else:
        file1 = base_path_3 + "" + end_day + "-ext_with_weather_" + "0301-0531_0801-0410" + "_" + str(
            False) + "_caiyun.csv"
        file5 = base_path_3 + "" + end_day + "-xgboost_weather" + "0301-0531_0801-0410" + "_" + str(
            True) + "_caiyun.csv"
        file6 = base_path_3 + "" + end_day + "-xgboost_weather" + "2017_0101-2018_0410_less" + "_" + str(
            False) + "_caiyun.csv"
        file7 = base_path_3 + "xgboost_ensemble_caiyun.csv"
    ensemble(file5, file6, file7, a=0.2)
    if caiyun == False:
        file12 = base_path_3 + end_day + "_ensemble_all_1.csv"
    else:
        file12 = base_path_3 + end_day + "_ensemble_all_1_caiyun.csv"
    ensemble(file1, file7, file12, a=0.7)


def get_ans_2(end_day, caiyun=False):
    if caiyun == False:
        file1 = base_path_3 + "" + end_day + "-ext_with_weather_" + "0301-0531_0801-0410" + "_" + str(
            False) + ".csv"
        file5 = base_path_3 + "" + end_day + "-xgboost_weather" + "2017_0101-2018_0410_less" + "_" + str(
            False) + ".csv"
        file6 = base_path_3 + "" + end_day + "-xgboost_weather" + "0301-0531_0801-0410" + "_" + str(
            False) + ".csv"
        file7 = base_path_3 + end_day + "_xgboost_ensemble.csv"
    else:
        file1 = base_path_3 + "" + end_day + "-ext_with_weather_" + "0301-0531_0801-0410" + "_" + str(
            False) + "_caiyun.csv"
        file5 = base_path_3 + "" + end_day + "-xgboost_weather" + "2017_0101-2018_0410_less" + "_" + str(
            False) + "_caiyun.csv"
        file6 = base_path_3 + "" + end_day + "-xgboost_weather" + "0301-0531_0801-0410" + "_" + str(
            False) + "_caiyun.csv"
        file7 = base_path_3 + end_day + "_xgboost_ensemble_caiyun.csv"
    ensemble(file5, file6, file7, a=0.3)
    if caiyun == False:
        file12 = base_path_3 + end_day + "_ensemble_all_2.csv"
    else:
        file12 = base_path_3 + end_day + "_ensemble_all_2_caiyun.csv"
    ensemble(file1, file7, file12, a=0.6)


def get_ans(end_day="2018-05-08", caiyun=False):
    # get_ans_1(end_day, caiyun=caiyun)
    # get_ans_2(end_day, caiyun=caiyun)
    # if caiyun == False:
    #     file13 = base_path_3 + end_day + "_ensemble_all_1.csv"
    #     file14 = base_path_3 + end_day + "_ensemble_all_2.csv"
    #     file15 = base_path_3 + end_day + "_ensemble_all_zhoujie.csv"
    # else:
    #     file13 = base_path_3 + end_day + "_ensemble_all_1_caiyun.csv"
    #     file14 = base_path_3 + end_day + "_ensemble_all_2_caiyun.csv"
    #     file15 = base_path_3 + end_day + "_ensemble_all_zhoujie_caiyun.csv"
    # ensemble(file13, file14, file15, a=0.4)

    if caiyun == False:
        file1 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + ".csv"
        file2 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + ".csv"
        file3 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "2017_0101-2018_0410_less" + "_" + str(
            False) + ".csv"
        file4 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "2017_0101-2018_0410_less" + "_" + str(
            False) + ".csv"
        file5 = base_path_3 + end_day + "lightgbm_mean_ensemble.csv"
        file6 = base_path_3 + end_day + "lightgbm_median_ensemble.csv"
    else:
        file1 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + "_caiyun.csv"
        file2 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + "_caiyun.csv"
        file3 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "2017_0101-2018_0410_less" + "_" + str(
            False) + "_caiyun.csv"
        file4 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "2017_0101-2018_0410_less" + "_" + str(
            False) + "_caiyun.csv"
        file5 = base_path_3 + end_day + "lightgbm_mean_ensemble_caiyun.csv"
        file6 = base_path_3 + end_day + "lightgbm_median_ensemble_caiyun.csv"

    ensemble_median(file1, file2, file3, file4, file5, file6)

    if caiyun == False:
        file1 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + ".csv"
        file2 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + ".csv"
        file3 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "2017_0101-2018_0429_less" + "_" + str(
            False) + ".csv"
        file4 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "2017_0101-2018_0429_less" + "_" + str(
            False) + ".csv"
        file5 = base_path_3 + end_day + "lightgbm_mean_ensemble_0429.csv"
        file6 = base_path_3 + end_day + "lightgbm_median_ensemble_0429.csv"
    else:
        file1 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + "_caiyun.csv"
        file2 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + "_caiyun.csv"
        file3 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "2017_0101-2018_0429_less" + "_" + str(
            False) + "_caiyun.csv"
        file4 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "2017_0101-2018_0429_less" + "_" + str(
            False) + "_caiyun.csv"
        file5 = base_path_3 + end_day + "lightgbm_mean_ensemble_0429_caiyun.csv"
        file6 = base_path_3 + end_day + "lightgbm_median_ensemble_0429_caiyun.csv"

    ensemble_median(file1, file2, file3, file4, file5, file6)

    if caiyun == False:
        file1 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + ".csv"
        file2 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + ".csv"
        file3 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "3" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + ".csv"
        file4 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "4" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + ".csv"
        file5 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "2017_0101-2018_0410_less" + "_" + str(
            False) + ".csv"
        file6 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "2017_0101-2018_0410_less" + "_" + str(
            False) + ".csv"
        file7 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "3" + "_" + "2017_0101-2018_0410_less" + "_" + str(
            False) + ".csv"
        file8 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "4" + "_" + "2017_0101-2018_0410_less" + "_" + str(
            False) + ".csv"
        file9 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "2017_0101-2018_0429_less" + "_" + str(
            False) + ".csv"
        file10 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "2017_0101-2018_0429_less" + "_" + str(
            False) + ".csv"
        file11 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "3" + "_" + "2017_0101-2018_0429_less" + "_" + str(
            False) + ".csv"
        file12 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "4" + "_" + "2017_0101-2018_0429_less" + "_" + str(
            False) + ".csv"
        file13 = base_path_3 + end_day + "lightgbm_mean_ensemble.csv"
        file14 = base_path_3 + end_day + "lightgbm_median_ensemble.csv"
    else:
        file1 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + "_caiyun.csv"
        file2 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + "_caiyun.csv"
        file3 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "3" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + "_caiyun.csv"
        file4 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "4" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + "_caiyun.csv"
        file5 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "2017_0101-2018_0410_less" + "_" + str(
            False) + "_caiyun.csv"
        file6 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "2017_0101-2018_0410_less" + "_" + str(
            False) + "_caiyun.csv"
        file7 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "3" + "_" + "2017_0101-2018_0410_less" + "_" + str(
            False) + "_caiyun.csv"
        file8 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "4" + "_" + "2017_0101-2018_0410_less" + "_" + str(
            False) + "_caiyun.csv"
        file9 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "2017_0101-2018_0429_less" + "_" + str(
            False) + "_caiyun.csv"
        file10 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "2017_0101-2018_0429_less" + "_" + str(
            False) + "_caiyun.csv"
        file11 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "3" + "_" + "2017_0101-2018_0429_less" + "_" + str(
            False) + "_caiyun.csv"
        file12 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "4" + "_" + "2017_0101-2018_0429_less" + "_" + str(
            False) + "_caiyun.csv"
        file13 = base_path_3 + end_day + "lightgbm_mean_ensemble_caiyun.csv"
        file14 = base_path_3 + end_day + "lightgbm_median_ensemble_caiyun.csv"

    ensemble_medians([file1, file2, file5, file6], file13, file14)

    if caiyun == False:
        file24 = base_path_3 + "" + end_day + "_weight_mean_0410.csv"
        file25 = base_path_3 + "" + end_day + "_weight_mean_0429.csv"
        file26 = base_path_3 + "" + end_day + "_weight_mean_0410_0429.csv"
    else:
        file24 = base_path_3 + "" + end_day + "_weight_mean_0410_caiyun.csv"
        file25 = base_path_3 + "" + end_day + "_weight_mean_0429_caiyun.csv"
        file26 = base_path_3 + "" + end_day + "_weight_mean_0410_0429_caiyun.csv"
    ensemble_medians_with_weight([file5, file7, file8], [0.6, 0.4, 0.0], file24)
    ensemble_medians_with_weight([file9, file11, file12], [0.5, 0.3, 0.2], file25)
    ensemble_medians_with_weight([file13, file14], [0.4, 0.6], file26)

    if caiyun == False:
        file27 = base_path_3 + "" + end_day + "_mean_0410.csv"
        file28 = base_path_3 + "" + end_day + "_median_0410.csv"
    else:
        file27 = base_path_3 + "" + end_day + "_mean_0410_caiyun.csv"
        file28 = base_path_3 + "" + end_day + "_median_0410_caiyun.csv"
    ensemble_medians([file5, file7, file8], file27, file28)

    if caiyun == False:
        file15 = base_path_3 + end_day + "lightgbm_mean_ensemble_6.csv"
        file16 = base_path_3 + end_day + "lightgbm_median_ensemble_6.csv"
    else:
        file15 = base_path_3 + end_day + "lightgbm_mean_ensemble_6_caiyun.csv"
        file16 = base_path_3 + end_day + "lightgbm_median_ensemble_6_caiyun.csv"
    ensemble_medians([file1, file2, file3, file4, file5, file6, file7, file8], file15, file16)

    if caiyun == False:
        file17 = base_path_3 + end_day + "lightgbm_mean_ensemble_29_6.csv"
        file18 = base_path_3 + end_day + "lightgbm_median_ensemble_29_6.csv"
    else:
        file17 = base_path_3 + end_day + "lightgbm_mean_ensemble_29_6_caiyun.csv"
        file18 = base_path_3 + end_day + "lightgbm_median_ensemble_29_6_caiyun.csv"
    ensemble_medians([file1, file2, file3, file4, file9, file10, file11, file12], file17, file18)

    if caiyun == False:
        file19 = base_path_3 + end_day + "lightgbm_ensemble_mean_4.csv"
        file20 = base_path_3 + end_day + "lightgbm_ensemble_median_4.csv"
    else:
        file19 = base_path_3 + end_day + "lightgbm_ensemble_mean_4_caiyun.csv"
        file20 = base_path_3 + end_day + "lightgbm_ensemble_median_4_caiyun.csv"
    ensemble_medians([file5, file7, file9, file11], file19, file20)

    if caiyun == False:
        file21 = base_path_3 + end_day + "lightgbm_ensemble_mean_2.csv"
        file22 = base_path_3 + end_day + "lightgbm_ensemble_median_2.csv"
    else:
        file21 = base_path_3 + end_day + "lightgbm_ensemble_mean_2_caiyun.csv"
        file22 = base_path_3 + end_day + "lightgbm_ensemble_median_2_caiyun.csv"
    ensemble_medians([file9, file11], file21, file22)

    if caiyun == False:
        file23 = base_path_3 + end_day + "lightgbm_ensemble_mean_weight.csv"
    else:
        file23 = base_path_3 + end_day + "lightgbm_ensemble_mean_weight_caiyun.csv"
    ensemble_medians_with_weight([file12, file8], [0.3, 0.7], file23)

    if caiyun == False:
        file1 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "2017_0101-2018_0429_clean" + "_" + str(
            False) + ".csv"
        file2 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "2017_0101-2018_0429_clean" + "_" + str(
            False) + ".csv"
        file3 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "3" + "_" + "2017_0101-2018_0429_clean" + "_" + str(
            False) + ".csv"
        file4 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "4" + "_" + "2017_0101-2018_0429_clean" + "_" + str(
            False) + ".csv"

        file24 = base_path_3 + end_day + "lightgbm_ensemble_mean__clean_4.csv"
        file25 = base_path_3 + end_day + "lightgbm_ensemble_median_clean_4.csv"
    else:
        file1 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "2017_0101-2018_0429_clean" + "_" + str(
            False) + "_caiyun.csv"
        file2 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "2017_0101-2018_0429_clean" + "_" + str(
            False) + "_caiyun.csv"
        file3 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "3" + "_" + "2017_0101-2018_0429_clean" + "_" + str(
            False) + "_caiyun.csv"
        file4 = base_path_3 + "" + end_day + "-lightgbm_weather_params_" + "4" + "_" + "2017_0101-2018_0429_clean" + "_" + str(
            False) + "_caiyun.csv"

        file24 = base_path_3 + end_day + "lightgbm_ensemble_mean__clean_4_caiyun.csv"
        file25 = base_path_3 + end_day + "lightgbm_ensemble_median_clean_4_caiyun.csv"
    # ensemble_medians([file1, file2, file3, file4], file24, file25)
    if caiyun == False:
        file26 = base_path_3 + end_day + "lightgbm_ensemble_mean_clean_2.csv"
        file27 = base_path_3 + end_day + "lightgbm_ensemble_median_clean_2.csv"
    else:
        file26 = base_path_3 + end_day + "lightgbm_ensemble_mean_clean_2_caiyun.csv"
        file27 = base_path_3 + end_day + "lightgbm_ensemble_median_clean_2_caiyun.csv"
    # ensemble_medians([file1, file3], file26, file27)

    # file19 = base_path_3 + "" + end_day + "-lightgbm_weather_log_params_" + "1" + "_" + "0301-0531_0801-0410" + "_" + str(
    #     False) + ".csv"
    # file20 = base_path_3 + "" + end_day + "-lightgbm_weather_log_params_" + "2" + "_" + "0301-0531_0801-0410" + "_" + str(
    #     False) + ".csv"
    # file21 = base_path_3 + "" + end_day + "-lightgbm_weather_log_params_" + "3" + "_" + "0301-0531_0801-0410" + "_" + str(
    #     False) + ".csv"
    # # file22 = base_path_3 + "" + end_day + "-lightgbm_weather_log_params_" + "4" + "_" + "0301-0531_0801-0410" + "_" + str(
    # #     False) + ".csv"
    # file23 = base_path_3 + "" + end_day + "-lightgbm_weather_log_params_" + "1" + "_" + "2017_0101-2018_0410_less" + "_" + str(
    #     False) + ".csv"
    # file24 = base_path_3 + "" + end_day + "-lightgbm_weather_log_params_" + "2" + "_" + "2017_0101-2018_0410_less" + "_" + str(
    #     False) + ".csv"
    # file25 = base_path_3 + "" + end_day + "-lightgbm_weather_log_params_" + "3" + "_" + "2017_0101-2018_0410_less" + "_" + str(
    #     False) + ".csv"
    # file26 = base_path_3 + "" + end_day + "-lightgbm_weather_log_params_" + "4" + "_" + "2017_0101-2018_0410_less" + "_" + str(
    #     False) + ".csv"

    # file27 = base_path_3 + "" + end_day + "-lightgbm_log_mean_ensemble.csv"
    # file28 = base_path_3 + "" + end_day + "-lightgbm_log_median_ensemble.csv"
    # ensemble_medians(
    #     [file1, file2, file3, file4, file5, file6, file7, file8, file19, file20, file21, file23, file24, file25,
    #      file26], file27, file28)


def get_ans_latter(end_day="2018-05-11", caiyun=False):
    get_ans_1(end_day, caiyun=caiyun)
    get_ans_2(end_day, caiyun=caiyun)
    if caiyun == False:
        file13 = base_path_3 + end_day + "_ensemble_all_1.csv"
        file14 = base_path_3 + end_day + "_ensemble_all_2.csv"
        file15 = base_path_3 + end_day + "_ensemble_all_zhoujie.csv"
    else:
        file13 = base_path_3 + end_day + "_ensemble_all_1_caiyun.csv"
        file14 = base_path_3 + end_day + "_ensemble_all_2_caiyun.csv"
        file15 = base_path_3 + end_day + "_ensemble_all_zhoujie_caiyun.csv"
    ensemble(file13, file14, file15, a=0.4)


def ensemble_mean():
    data_base = "../output/"
    df1 = pd.read_csv(data_base + 'friend/sub20180502_060127.csv')
    df2 = pd.read_csv(data_base + '2018-05-01-ext_with_weather_0301-0531_0801-0410_False.csv')
    df3 = pd.read_csv(data_base + '2018-05-01-xgboost_weather0301-0531_0801-0410_False.csv')
    df4 = pd.read_csv(data_base + 'friend/res2018-05-01.csv')
    df1.columns = ['test_id', 'PM2.5_df1', 'PM10_df1', 'O3_df1']
    df2.columns = ['test_id', 'PM2.5_df2', 'PM10_df2', 'O3_df2']
    df3.columns = ['test_id', 'PM2.5_df3', 'PM10_df3', 'O3_df3']
    df4.columns = ['test_id', 'PM2.5_df4', 'PM10_df4', 'O3_df4']
    df = df1
    df = pd.merge(df, df2, on='test_id', how='left')
    df = pd.merge(df, df3, on='test_id', how='left')
    df = pd.merge(df, df4, on='test_id', how='left')

    # df.columns

    def en_mean(x):
        return np.mean([x[0], x[1], x[2], x[3]])

    def en_median(x):
        return np.median([x[0], x[1], x[2], x[3]])

    df['PM2.5'] = df[['PM2.5_df1', 'PM2.5_df2', 'PM2.5_df3', 'PM2.5_df4']].apply(lambda x: en_mean(x), axis=1)
    df['PM10'] = df[['PM10_df1', 'PM10_df2', 'PM10_df3', 'PM10_df4']].apply(lambda x: en_mean(x), axis=1)
    df['O3'] = df[['O3_df1', 'O3_df2', 'O3_df3', 'O3_df4']].apply(lambda x: en_mean(x), axis=1)
    df[['test_id', 'PM2.5', 'PM10', 'O3']].to_csv(data_base + 'four_result_mean0501.csv', index=False)

    df['PM2.5'] = df[['PM2.5_df1', 'PM2.5_df2', 'PM2.5_df3', 'PM2.5_df4']].apply(lambda x: en_median(x), axis=1)
    df['PM10'] = df[['PM10_df1', 'PM10_df2', 'PM10_df3', 'PM10_df4']].apply(lambda x: en_median(x), axis=1)
    df['O3'] = df[['O3_df1', 'O3_df2', 'O3_df3', 'O3_df4']].apply(lambda x: en_median(x), axis=1)
    df[['test_id', 'PM2.5', 'PM10', 'O3']].to_csv(data_base + 'four_result_median0501.csv', index=False)


def ensemble_mean_2():
    data_base = "../output/"
    df1 = pd.read_csv(data_base + 'friend/sub20180502_060127.csv')
    df2 = pd.read_csv(data_base + '2018-05-01-ext_with_weather_0301-0531_0801-0410_False.csv')
    df3 = pd.read_csv(data_base + '2018-05-01-xgboost_weather0301-0531_0801-0410_False.csv')
    df4 = pd.read_csv(data_base + 'friend/res2018-05-01.csv')
    df1.columns = ['test_id', 'PM2.5_df1', 'PM10_df1', 'O3_df1']
    df2.columns = ['test_id', 'PM2.5_df2', 'PM10_df2', 'O3_df2']
    df3.columns = ['test_id', 'PM2.5_df3', 'PM10_df3', 'O3_df3']
    df4.columns = ['test_id', 'PM2.5_df4', 'PM10_df4', 'O3_df4']
    df = df1
    df = pd.merge(df, df2, on='test_id', how='left')
    df = pd.merge(df, df3, on='test_id', how='left')
    df = pd.merge(df, df4, on='test_id', how='left')

    # df.columns

    def en_mean(x):
        return np.mean([x[0], x[1], x[2]])

    def en_median(x):
        return np.median([x[0], x[1], x[2]])

    df['PM2.5'] = df[['PM2.5_df1', 'PM2.5_df2', 'PM2.5_df3']].apply(lambda x: en_mean(x), axis=1)
    df['PM10'] = df[['PM10_df1', 'PM10_df2', 'PM10_df3']].apply(lambda x: en_mean(x), axis=1)
    df['O3'] = df[['O3_df1', 'O3_df2', 'O3_df3']].apply(lambda x: en_mean(x), axis=1)
    df[['test_id', 'PM2.5', 'PM10', 'O3']].to_csv(data_base + 'three_result_mean0501.csv', index=False)

    df['PM2.5'] = df[['PM2.5_df1', 'PM2.5_df2', 'PM2.5_df3']].apply(lambda x: en_median(x), axis=1)
    df['PM10'] = df[['PM10_df1', 'PM10_df2', 'PM10_df3']].apply(lambda x: en_median(x), axis=1)
    df['O3'] = df[['O3_df1', 'O3_df2', 'O3_df3']].apply(lambda x: en_median(x), axis=1)
    df[['test_id', 'PM2.5', 'PM10', 'O3']].to_csv(data_base + 'three_result_median0501.csv', index=False)


def ensemble_mean_3(file1, file2, file3, fileto1, fileto2):
    data_base = "../output/"
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    df1.columns = ['test_id', 'PM2.5_df1', 'PM10_df1', 'O3_df1']
    df2.columns = ['test_id', 'PM2.5_df2', 'PM10_df2', 'O3_df2']
    df3.columns = ['test_id', 'PM2.5_df3', 'PM10_df3', 'O3_df3']
    df = df1
    df = pd.merge(df, df2, on='test_id', how='left')
    df = pd.merge(df, df3, on='test_id', how='left')

    # df.columns

    def en_mean(x):
        return np.mean([x[0], x[1], x[2]])

    def en_median(x):
        return np.median([x[0], x[1], x[2]])

    df['PM2.5'] = df[['PM2.5_df1', 'PM2.5_df2', 'PM2.5_df3']].apply(lambda x: en_mean(x), axis=1)
    df['PM10'] = df[['PM10_df1', 'PM10_df2', 'PM10_df3']].apply(lambda x: en_mean(x), axis=1)
    df['O3'] = df[['O3_df1', 'O3_df2', 'O3_df3']].apply(lambda x: en_mean(x), axis=1)
    df[['test_id', 'PM2.5', 'PM10', 'O3']].to_csv(fileto1, index=False)

    df['PM2.5'] = df[['PM2.5_df1', 'PM2.5_df2', 'PM2.5_df3']].apply(lambda x: en_median(x), axis=1)
    df['PM10'] = df[['PM10_df1', 'PM10_df2', 'PM10_df3']].apply(lambda x: en_median(x), axis=1)
    df['O3'] = df[['O3_df1', 'O3_df2', 'O3_df3']].apply(lambda x: en_median(x), axis=1)
    df[['test_id', 'PM2.5', 'PM10', 'O3']].to_csv(fileto2, index=False)


def cal_ensemble_lightgbm():
    best_params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    min_value = 10000.0
    for t1 in range(11):
        w1 = t1 / 10.0
        for t2 in range(11):
            w2 = t2 / 10.0
            if w1 + w2 > 1.0:
                break
            for t3 in range(11):
                w3 = t3 / 10.0
                if w1 + w2 + w3 > 1.0:
                    break
                for t4 in range(11):
                    w4 = t4 / 10.0
                    if w1 + w2 + w3 + w4 > 1.0:
                        break
                    for t5 in range(11):
                        w5 = t5 / 10.0
                        if w1 + w2 + w3 + w4 + w5 > 1.0:
                            break
                        w6 = 1 - w1 - w2 - w3 - w4 - w5
                        score = 0.0
                        num = 0.0
                        for j in range(30, 35):
                            if j <= 30:
                                end_day = "2018-04-" + str(j)
                            else:
                                end_day = "2018-05-0" + str(j - 30)
                            file1 = base_path_3 + "test/" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "0301-0531_0801-0410" + "_" + str(
                                False) + ".csv"
                            file2 = base_path_3 + "test/" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "0301-0531_0801-0410" + "_" + str(
                                False) + ".csv"
                            file3 = base_path_3 + "test/" + end_day + "-lightgbm_weather_params_" + "3" + "_" + "0301-0531_0801-0410" + "_" + str(
                                False) + ".csv"
                            file4 = base_path_3 + "test/" + end_day + "-lightgbm_weather_params_" + "4" + "_" + "0301-0531_0801-0410" + "_" + str(
                                False) + ".csv"
                            file5 = base_path_3 + "test/" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "2017_0101-2018_0410_less" + "_" + str(
                                False) + ".csv"
                            file6 = base_path_3 + "test/" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "2017_0101-2018_0410_less" + "_" + str(
                                False) + ".csv"
                            file7 = base_path_3 + "test/" + end_day + "-lightgbm_weather_params_" + "3" + "_" + "2017_0101-2018_0410_less" + "_" + str(
                                False) + ".csv"
                            file8 = base_path_3 + "test/" + end_day + "-lightgbm_weather_params_" + "4" + "_" + "2017_0101-2018_0410_less" + "_" + str(
                                False) + ".csv"
                            file9 = base_path_3 + "test/" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "2017_0101-2018_0429_less" + "_" + str(
                                False) + ".csv"
                            file10 = base_path_3 + "test/" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "2017_0101-2018_0429_less" + "_" + str(
                                False) + ".csv"
                            file11 = base_path_3 + "test/" + end_day + "-lightgbm_weather_params_" + "3" + "_" + "2017_0101-2018_0429_less" + "_" + str(
                                False) + ".csv"
                            file12 = base_path_3 + "test/" + end_day + "-lightgbm_weather_params_" + "4" + "_" + "2017_0101-2018_0429_less" + "_" + str(
                                False) + ".csv"
                            file13 = base_path_3 + "test/" + end_day + "_weight_mean.csv"
                            ensemble_medians_with_weight([file9, file11, file12, file5, file7, file8],
                                                         [w1, w2, w3, w4, w5, w6], file13)
                            score13, _, _ = test(end_day, file13)
                            score += score13
                            num += 1.0
                        avg_score = score / num
                        if avg_score < min_value:
                            min_value = avg_score
                            best_params = [w1, w2, w3, w4, w5, w6]
                            print min_value, best_params
    print best_params, min_value

    # score = np.zeros(18)
    # num = 0.0
    # for j in range(29, 35):
    #     if j <= 30:
    #         end_day = "2018-04-" + str(j)
    #     else:
    #         end_day = "2018-05-0" + str(j - 30)
    #     file1 = base_path_3 + "test/" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "0301-0531_0801-0410" + "_" + str(
    #         False) + ".csv"
    #     file2 = base_path_3 + "test/" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "0301-0531_0801-0410" + "_" + str(
    #         False) + ".csv"
    #     file3 = base_path_3 + "test/" + end_day + "-lightgbm_weather_params_" + "3" + "_" + "0301-0531_0801-0410" + "_" + str(
    #         False) + ".csv"
    #     file4 = base_path_3 + "test/" + end_day + "-lightgbm_weather_params_" + "4" + "_" + "0301-0531_0801-0410" + "_" + str(
    #         False) + ".csv"
    #     file5 = base_path_3 + "test/" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "2017_0101-2018_0410_less" + "_" + str(
    #         False) + ".csv"
    #     file6 = base_path_3 + "test/" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "2017_0101-2018_0410_less" + "_" + str(
    #         False) + ".csv"
    #     file7 = base_path_3 + "test/" + end_day + "-lightgbm_weather_params_" + "3" + "_" + "2017_0101-2018_0410_less" + "_" + str(
    #         False) + ".csv"
    #     file8 = base_path_3 + "test/" + end_day + "-lightgbm_weather_params_" + "4" + "_" + "2017_0101-2018_0410_less" + "_" + str(
    #         False) + ".csv"
    #     file9 = base_path_3 + "test/" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "2017_0101-2018_0429_less" + "_" + str(
    #         False) + ".csv"
    #     file10 = base_path_3 + "test/" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "2017_0101-2018_0429_less" + "_" + str(
    #         False) + ".csv"
    #     file11 = base_path_3 + "test/" + end_day + "-lightgbm_weather_params_" + "3" + "_" + "2017_0101-2018_0429_less" + "_" + str(
    #         False) + ".csv"
    #     file12 = base_path_3 + "test/" + end_day + "-lightgbm_weather_params_" + "4" + "_" + "2017_0101-2018_0429_less" + "_" + str(
    #         False) + ".csv"
    #     file13 = base_path_3 + end_day + "lightgbm_mean_ensemble.csv"
    #     file14 = base_path_3 + end_day + "lightgbm_median_ensemble.csv"
    #     ensemble_medians([file1, file2, file5, file6], file13, file14)
    #
    #     file15 = base_path_3 + end_day + "lightgbm_mean_ensemble_6.csv"
    #     file16 = base_path_3 + end_day + "lightgbm_median_ensemble_6.csv"
    #     ensemble_medians([file1, file2, file3, file4, file5, file6, file7, file8], file15, file16)
    #
    #     file17 = base_path_3 + end_day + "lightgbm_mean_ensemble_2.csv"
    #     file18 = base_path_3 + end_day + "lightgbm_median_ensemble_2.csv"
    #     ensemble_medians([file1, file3, file5, file7], file17, file18)
    #
    #     score1, _, _ = test(end_day, file1)
    #     score2, _, _ = test(end_day, file2)
    #     score3, _, _ = test(end_day, file3)
    #     score4, _, _ = test(end_day, file4)
    #     score5, _, _ = test(end_day, file5)
    #     score6, _, _ = test(end_day, file6)
    #     score7, _, _ = test(end_day, file7)
    #     score8, _, _ = test(end_day, file8)
    #     score9, _, _ = test(end_day, file9)
    #     score10, _, _ = test(end_day, file10)
    #     # score11, _, _ = test(end_day, file11)
    #     # score12, _, _ = test(end_day, file12)
    #     score13, _, _ = test(end_day, file13)
    #     score14, _, _ = test(end_day, file14)
    #     score15, _, _ = test(end_day, file15)
    #     score16, _, _ = test(end_day, file16)
    #     score17, _, _ = test(end_day, file17)
    #     score18, _, _ = test(end_day, file18)
    #     score_now = np.array(
    #         [score1, score2, score3, score4, score5, score6, score7, score8, score9, score10, 0, 0, score13, score14,
    #          score15, score16, score17, score18])
    #     score += score_now
    #     print "score: ", score_now
    #     num += 1.0
    # avg_score = score / num
    # print "avg_score: ", avg_score


def ensemble_medians(filenames, fileto1, fileto2):
    df = None
    for i in range(len(filenames)):
        df_tmp = pd.read_csv(filenames[i])
        df_tmp.columns = ['test_id', 'PM2.5_df' + str(i + 1), 'PM10_df' + str(i + 1), 'O3_df' + str(i + 1)]
        if df is None:
            df = df_tmp
        else:
            df = pd.merge(df, df_tmp, on='test_id', how='left')

    def en_mean(x):
        return np.mean([x[i] for i in range(len(filenames))])

    def en_median(x):
        return np.median([x[i] for i in range(len(filenames))])

    df['PM2.5'] = df[['PM2.5_df' + str(i) for i in range(1, len(filenames) + 1)]].apply(lambda x: en_mean(x), axis=1)
    df['PM10'] = df[['PM10_df' + str(i) for i in range(1, len(filenames) + 1)]].apply(lambda x: en_mean(x), axis=1)
    df['O3'] = df[['O3_df' + str(i) for i in range(1, len(filenames) + 1)]].apply(lambda x: en_mean(x), axis=1)
    df[['test_id', 'PM2.5', 'PM10', 'O3']].to_csv(fileto1, index=False)

    df['PM2.5'] = df[['PM2.5_df' + str(i) for i in range(1, len(filenames) + 1)]].apply(lambda x: en_median(x), axis=1)
    df['PM10'] = df[['PM10_df' + str(i) for i in range(1, len(filenames) + 1)]].apply(lambda x: en_median(x), axis=1)
    df['O3'] = df[['O3_df' + str(i) for i in range(1, len(filenames) + 1)]].apply(lambda x: en_median(x), axis=1)
    df[['test_id', 'PM2.5', 'PM10', 'O3']].to_csv(fileto2, index=False)


def ensemble_medians_with_weight(filenames, weight, fileto1):
    df = None
    for i in range(len(filenames)):
        df_tmp = pd.read_csv(filenames[i])
        df_tmp.columns = ['test_id', 'PM2.5_df' + str(i + 1), 'PM10_df' + str(i + 1), 'O3_df' + str(i + 1)]
        if df is None:
            df = df_tmp
        else:
            df = pd.merge(df, df_tmp, on='test_id', how='left')

    def en_weight_mean(x):
        return np.sum([x[i] * weight[i] for i in range(len(filenames))])

    df['PM2.5'] = df[['PM2.5_df' + str(i) for i in range(1, len(filenames) + 1)]].apply(lambda x: en_weight_mean(x),
                                                                                        axis=1)
    df['PM10'] = df[['PM10_df' + str(i) for i in range(1, len(filenames) + 1)]].apply(lambda x: en_weight_mean(x),
                                                                                      axis=1)
    df['O3'] = df[['O3_df' + str(i) for i in range(1, len(filenames) + 1)]].apply(lambda x: en_weight_mean(x), axis=1)
    df[['test_id', 'PM2.5', 'PM10', 'O3']].to_csv(fileto1, index=False)


def ensemble_median(filename1, filename2, filename3, filename4, fileto1, fileto2):
    data_base = "../output/"
    df1 = pd.read_csv(filename1)
    df2 = pd.read_csv(filename2)
    df3 = pd.read_csv(filename3)
    df4 = pd.read_csv(filename4)
    df1.columns = ['test_id', 'PM2.5_df1', 'PM10_df1', 'O3_df1']
    df2.columns = ['test_id', 'PM2.5_df2', 'PM10_df2', 'O3_df2']
    df3.columns = ['test_id', 'PM2.5_df3', 'PM10_df3', 'O3_df3']
    df4.columns = ['test_id', 'PM2.5_df4', 'PM10_df4', 'O3_df4']
    df = df1
    df = pd.merge(df, df2, on='test_id', how='left')
    df = pd.merge(df, df3, on='test_id', how='left')
    df = pd.merge(df, df4, on='test_id', how='left')

    # df.columns

    def en_mean(x):
        return np.mean([x[0], x[1], x[2], x[3]])

    def en_median(x):
        return np.median([x[0], x[1], x[2], x[3]])

    df['PM2.5'] = df[['PM2.5_df1', 'PM2.5_df2', 'PM2.5_df3', 'PM2.5_df4']].apply(lambda x: en_mean(x), axis=1)
    df['PM10'] = df[['PM10_df1', 'PM10_df2', 'PM10_df3', 'PM10_df4']].apply(lambda x: en_mean(x), axis=1)
    df['O3'] = df[['O3_df1', 'O3_df2', 'O3_df3', 'O3_df4']].apply(lambda x: en_mean(x), axis=1)
    df[['test_id', 'PM2.5', 'PM10', 'O3']].to_csv(fileto1, index=False)

    df['PM2.5'] = df[['PM2.5_df1', 'PM2.5_df2', 'PM2.5_df3', 'PM2.5_df4']].apply(lambda x: en_median(x), axis=1)
    df['PM10'] = df[['PM10_df1', 'PM10_df2', 'PM10_df3', 'PM10_df4']].apply(lambda x: en_median(x), axis=1)
    df['O3'] = df[['O3_df1', 'O3_df2', 'O3_df3', 'O3_df4']].apply(lambda x: en_median(x), axis=1)
    df[['test_id', 'PM2.5', 'PM10', 'O3']].to_csv(fileto2, index=False)


def cal_ans(type, feature_first=False):
    print type
    for i in range(11, 28):
        start_day = "2018-04-" + str(i)
        end_day = "2018-04-" + str(i)
        ans_file = base_path_3 + "test/" + end_day + "-lightgbm_weather" + type + "_" + str(feature_first) + ".csv"
        score, _, _ = test(end_day, ans_file)
        print score


def cal_ensemble_lightgbm_real():
    best_params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    min_value = 10000.0
    for t1 in range(11):
        w1 = t1 / 10.0
        for t2 in range(11):
            w2 = t2 / 10.0
            if w1 + w2 > 1.0:
                break
            for t3 in range(11):
                w3 = t3 / 10.0
                if w1 + w2 + w3 > 1.0:
                    break
                for t4 in range(11):
                    w4 = t4 / 10.0
                    if w1 + w2 + w3 + w4 > 1.0:
                        break
                    for t5 in range(11):
                        w5 = t5 / 10.0
                        if w1 + w2 + w3 + w4 + w5 > 1.0:
                            break
                        w6 = 1 - w1 - w2 - w3 - w4 - w5
                        score = 0.0
                        num = 0.0
                        for j in range(4, 10):
                            end_day = "2018-05-%02d" % j
                            file1 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "0301-0531_0801-0410" + "_" + str(
                                False) + ".csv"
                            file2 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "0301-0531_0801-0410" + "_" + str(
                                False) + ".csv"
                            file3 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "3" + "_" + "0301-0531_0801-0410" + "_" + str(
                                False) + ".csv"
                            file4 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "4" + "_" + "0301-0531_0801-0410" + "_" + str(
                                False) + ".csv"
                            file5 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "2017_0101-2018_0410_less" + "_" + str(
                                False) + ".csv"
                            file6 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "2017_0101-2018_0410_less" + "_" + str(
                                False) + ".csv"
                            file7 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "3" + "_" + "2017_0101-2018_0410_less" + "_" + str(
                                False) + ".csv"
                            file8 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "4" + "_" + "2017_0101-2018_0410_less" + "_" + str(
                                False) + ".csv"
                            file9 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "2017_0101-2018_0429_less" + "_" + str(
                                False) + ".csv"
                            file10 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "2017_0101-2018_0429_less" + "_" + str(
                                False) + ".csv"
                            file11 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "3" + "_" + "2017_0101-2018_0429_less" + "_" + str(
                                False) + ".csv"
                            file12 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "4" + "_" + "2017_0101-2018_0429_less" + "_" + str(
                                False) + ".csv"
                            file13 = base_path_3 + "old/" + end_day + "_weight_mean.csv"
                            ensemble_medians_with_weight([file5, file7, file8, file9, file11, file12],
                                                         [w1, w2, w3, w4, w5, w6], file13)
                            score13, _, _ = test(end_day, file13)
                            score += score13
                            num += 1.0
                        avg_score = score / num
                        if avg_score < min_value:
                            min_value = avg_score
                            best_params = [w1, w2, w3, w4, w5, w6]
                            print min_value, best_params
    print best_params, min_value


def get_ans_history(end_day="2018-05-08", caiyun=False):
    if caiyun == False:
        file1 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + ".csv"
        file2 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + ".csv"
        file3 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "2017_0101-2018_0410_less" + "_" + str(
            False) + ".csv"
        file4 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "2017_0101-2018_0410_less" + "_" + str(
            False) + ".csv"
        file5 = base_path_3 + end_day + "lightgbm_mean_ensemble.csv"
        file6 = base_path_3 + end_day + "lightgbm_median_ensemble.csv"
    else:
        file1 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + "_caiyun.csv"
        file2 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + "_caiyun.csv"
        file3 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "2017_0101-2018_0410_less" + "_" + str(
            False) + "_caiyun.csv"
        file4 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "2017_0101-2018_0410_less" + "_" + str(
            False) + "_caiyun.csv"
        file5 = base_path_3 + end_day + "lightgbm_mean_ensemble_caiyun.csv"
        file6 = base_path_3 + end_day + "lightgbm_median_ensemble_caiyun.csv"

    ensemble_median(file1, file2, file3, file4, file5, file6)

    if caiyun == False:
        file1 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + ".csv"
        file2 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + ".csv"
        file3 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "2017_0101-2018_0429_less" + "_" + str(
            False) + ".csv"
        file4 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "2017_0101-2018_0429_less" + "_" + str(
            False) + ".csv"
        file5 = base_path_3 + end_day + "lightgbm_mean_ensemble_0429.csv"
        file6 = base_path_3 + end_day + "lightgbm_median_ensemble_0429.csv"
    else:
        file1 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + "_caiyun.csv"
        file2 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + "_caiyun.csv"
        file3 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "2017_0101-2018_0429_less" + "_" + str(
            False) + "_caiyun.csv"
        file4 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "2017_0101-2018_0429_less" + "_" + str(
            False) + "_caiyun.csv"
        file5 = base_path_3 + end_day + "lightgbm_mean_ensemble_0429_caiyun.csv"
        file6 = base_path_3 + end_day + "lightgbm_median_ensemble_0429_caiyun.csv"

    ensemble_median(file1, file2, file3, file4, file5, file6)

    if caiyun == False:
        file1 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + ".csv"
        file2 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + ".csv"
        file3 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "3" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + ".csv"
        file4 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "4" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + ".csv"
        file5 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "2017_0101-2018_0410_less" + "_" + str(
            False) + ".csv"
        file6 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "2017_0101-2018_0410_less" + "_" + str(
            False) + ".csv"
        file7 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "3" + "_" + "2017_0101-2018_0410_less" + "_" + str(
            False) + ".csv"
        file8 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "4" + "_" + "2017_0101-2018_0410_less" + "_" + str(
            False) + ".csv"
        file9 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "2017_0101-2018_0429_less" + "_" + str(
            False) + ".csv"
        file10 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "2017_0101-2018_0429_less" + "_" + str(
            False) + ".csv"
        file11 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "3" + "_" + "2017_0101-2018_0429_less" + "_" + str(
            False) + ".csv"
        file12 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "4" + "_" + "2017_0101-2018_0429_less" + "_" + str(
            False) + ".csv"
        file13 = base_path_3 + end_day + "lightgbm_mean_ensemble.csv"
        file14 = base_path_3 + end_day + "lightgbm_median_ensemble.csv"
    else:
        file1 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + "_caiyun.csv"
        file2 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + "_caiyun.csv"
        file3 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "3" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + "_caiyun.csv"
        file4 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "4" + "_" + "0301-0531_0801-0410" + "_" + str(
            False) + "_caiyun.csv"
        file5 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "2017_0101-2018_0410_less" + "_" + str(
            False) + "_caiyun.csv"
        file6 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "2017_0101-2018_0410_less" + "_" + str(
            False) + "_caiyun.csv"
        file7 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "3" + "_" + "2017_0101-2018_0410_less" + "_" + str(
            False) + "_caiyun.csv"
        file8 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "4" + "_" + "2017_0101-2018_0410_less" + "_" + str(
            False) + "_caiyun.csv"
        file9 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "2017_0101-2018_0429_less" + "_" + str(
            False) + "_caiyun.csv"
        file10 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "2017_0101-2018_0429_less" + "_" + str(
            False) + "_caiyun.csv"
        file11 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "3" + "_" + "2017_0101-2018_0429_less" + "_" + str(
            False) + "_caiyun.csv"
        file12 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "4" + "_" + "2017_0101-2018_0429_less" + "_" + str(
            False) + "_caiyun.csv"
        file13 = base_path_3 + end_day + "lightgbm_mean_ensemble_caiyun.csv"
        file14 = base_path_3 + end_day + "lightgbm_median_ensemble_caiyun.csv"

    ensemble_medians([file1, file2, file5, file6], file13, file14)

    if caiyun == False:
        file24 = base_path_3 + "old/" + end_day + "_weight_mean_0410.csv"
        file25 = base_path_3 + "old/" + end_day + "_weight_mean_0429.csv"
        file26 = base_path_3 + "old/" + end_day + "_weight_mean_0410_0429.csv"
    else:
        file24 = base_path_3 + "old/" + end_day + "_weight_mean_0410_caiyun.csv"
        file25 = base_path_3 + "old/" + end_day + "_weight_mean_0429_caiyun.csv"
        file26 = base_path_3 + "old/" + end_day + "_weight_mean_0410_0429_caiyun.csv"
    ensemble_medians_with_weight([file5, file7, file8], [0.6, 0.4, 0.0], file24)
    ensemble_medians_with_weight([file9, file11, file12], [0.5, 0.3, 0.2], file25)
    ensemble_medians_with_weight([file13, file14], [0.4, 0.6], file26)

    if caiyun == False:
        file27 = base_path_3 + "old/" + end_day + "_mean_0410.csv"
        file28 = base_path_3 + "old/" + end_day + "_median_0410.csv"
    else:
        file27 = base_path_3 + "old/" + end_day + "_mean_0410_caiyun.csv"
        file28 = base_path_3 + "old/" + end_day + "_median_0410_caiyun.csv"
    ensemble_medians([file5, file7, file8], file27, file28)

    if caiyun == False:
        file15 = base_path_3 + "old/" + end_day + "lightgbm_mean_ensemble_6.csv"
        file16 = base_path_3 + "old/" + end_day + "lightgbm_median_ensemble_6.csv"
    else:
        file15 = base_path_3 + "old/" + end_day + "lightgbm_mean_ensemble_6_caiyun.csv"
        file16 = base_path_3 + "old/" + end_day + "lightgbm_median_ensemble_6_caiyun.csv"
    ensemble_medians([file1, file2, file3, file4, file5, file6, file7, file8], file15, file16)

    if caiyun == False:
        file17 = base_path_3 + "old/" + end_day + "lightgbm_mean_ensemble_29_6.csv"
        file18 = base_path_3 + "old/" + end_day + "lightgbm_median_ensemble_29_6.csv"
    else:
        file17 = base_path_3 + "old/" + end_day + "lightgbm_mean_ensemble_29_6_caiyun.csv"
        file18 = base_path_3 + "old/" + end_day + "lightgbm_median_ensemble_29_6_caiyun.csv"
    ensemble_medians([file1, file2, file3, file4, file9, file10, file11, file12], file17, file18)

    if caiyun == False:
        file19 = base_path_3 + "old/" + end_day + "lightgbm_ensemble_mean_4.csv"
        file20 = base_path_3 + "old/" + end_day + "lightgbm_ensemble_median_4.csv"
    else:
        file19 = base_path_3 + "old/" + end_day + "lightgbm_ensemble_mean_4_caiyun.csv"
        file20 = base_path_3 + "old/" + end_day + "lightgbm_ensemble_median_4_caiyun.csv"
    ensemble_medians([file5, file7, file9, file11], file19, file20)

    if caiyun == False:
        file21 = base_path_3 + "old/" + end_day + "lightgbm_ensemble_mean_2.csv"
        file22 = base_path_3 + "old/" + end_day + "lightgbm_ensemble_median_2.csv"
    else:
        file21 = base_path_3 + "old/" + end_day + "lightgbm_ensemble_mean_2_caiyun.csv"
        file22 = base_path_3 + "old/" + end_day + "lightgbm_ensemble_median_2_caiyun.csv"
    ensemble_medians([file9, file11], file21, file22)

    if caiyun == False:
        file23 = base_path_3 + "old/" + end_day + "lightgbm_ensemble_mean_weight.csv"
    else:
        file23 = base_path_3 + "old/" + end_day + "lightgbm_ensemble_mean_weight_caiyun.csv"
    ensemble_medians_with_weight([file12, file8], [0.3, 0.7], file23)


def change_attr(filename1, filename2, filename3):
    df1 = pd.read_csv(filename1)
    df2 = pd.read_csv(filename2)
    df1.columns = ['test_id', 'PM2.5_df1', 'PM10_df1', 'O3_df1']
    df2.columns = ['test_id', 'PM2.5_df2', 'PM10_df2', 'O3_df2']
    df = df1
    df = pd.merge(df, df2, on='test_id', how='left')

    df['PM2.5'] = df[['PM2.5_df1', 'PM2.5_df2']].apply(lambda x: x[1], axis=1)
    df['PM10'] = df[['PM10_df1', 'PM10_df2']].apply(lambda x: x[0], axis=1)
    df['O3'] = df[['O3_df1', 'O3_df2']].apply(lambda x: x[0], axis=1)
    df[['test_id', 'PM2.5', 'PM10', 'O3']].to_csv(filename3, index=False)

def change_attr_1(filename1, filename2, filename3):
    df1 = pd.read_csv(filename1)
    df2 = pd.read_csv(filename2)
    df1.columns = ['test_id', 'PM2.5_df1', 'PM10_df1', 'O3_df1']
    df2.columns = ['test_id', 'PM2.5_df2', 'PM10_df2', 'O3_df2']
    df = df1
    df = pd.merge(df, df2, on='test_id', how='left')

    df['PM2.5'] = df[['PM2.5_df1', 'PM2.5_df2']].apply(lambda x: x[1], axis=1)
    df['PM10'] = df[['PM10_df1', 'PM10_df2']].apply(lambda x: x[0], axis=1)
    df['O3'] = df[['O3_df1', 'O3_df2']].apply(lambda x: x[0], axis=1)
    df[['test_id', 'PM2.5', 'PM10', 'O3']].to_csv(filename3, index=False)


if __name__ == '__main__':
    # file1 = base_path_3 + "2018-05-18-lightgbm_weather_params_1_2017_0101-2018_0429_less_False.csv"
    # file2 = base_path_3 + "sub20180519_074905.csv"
    # file3 = base_path_3 + "2018-05-18-lightgbm_weather_params_1_2017_0101-2018_0429_less_False_piu_0.5.csv"
    # ensemble_medians_with_weight([file1, file2], [0.1, 0.9], file3)
    day = "27"
    base_path_4 = '../image/results/201805' + day + '/'
    file1 = base_path_4 + "2018-05-" + day + "lightgbm_ensemble_mean_4.csv"
    file2 = base_path_4 + "piu_091_lightgbm_weather_params_5_2017_0101-2018_0515_less_False_009.csv"
    file3 = base_path_4 + "2018-05-" + day + "test.csv"
    ensemble_medians_with_weight([file1, file2], [0.5, 0.5], file3)
    # file3 = base_path_4 + "2018-05-" + day + "lightgbm_ensemble_median_4_lightgbm_weather_params_5_change_PM25.csv"
    # change_attr(file2, file1, file3)

    # [0.0, 0.3, 0.7]
    # cal_ensemble_best_xgboost()
    # 1.0 0 0
    # cal_ensemble_best_ext_with_weather()
    # 1.0 0.0 0.0
    # cal_ensemble_best_ext_with_weather_three_metric()
    # 0.7 0.3 0.0
    # cal_ensemble_all()
    # get_ans(end_day='2018-05-12')
    # get_ans(end_day='2018-05-12', caiyun=True)
    # cal_ans(type="0301-0531_0801-0410", feature_first=False)
    # cal_ans(type="2017_0101-2018_0410_less", feature_first=False)
    # cal_ensemble_lightgbm()
    # cal_ensemble_lightgbm_real()
    # get_ans_latter(caiyun=True)
    # for i in range(4, 5):
    #     end_day = '2018-05-%02d' % i
    #     get_ans_history(end_day=end_day)
    # get_ans_history(end_day="2018-05-11", caiyun=True)
    # get_ans_history(end_day="2018-05-12", caiyun=True)
    # score = np.zeros(4)
    # num = 0.0
    # for j in range(10, 12):
    #     end_day = "2018-05-%02d" % j
    #     file1 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "0301-0531_0801-0410" + "_" + str(
    #         False) + ".csv"
    #     file2 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "0301-0531_0801-0410" + "_" + str(
    #         False) + ".csv"
    #     file3 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "3" + "_" + "0301-0531_0801-0410" + "_" + str(
    #         False) + ".csv"
    #     file4 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "4" + "_" + "0301-0531_0801-0410" + "_" + str(
    #         False) + ".csv"
    #     file5 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "2017_0101-2018_0410_less" + "_" + str(
    #         False) + ".csv"
    #     file6 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "2017_0101-2018_0410_less" + "_" + str(
    #         False) + ".csv"
    #     file7 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "3" + "_" + "2017_0101-2018_0410_less" + "_" + str(
    #         False) + ".csv"
    #     file8 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "4" + "_" + "2017_0101-2018_0410_less" + "_" + str(
    #         False) + ".csv"
    #     file9 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "1" + "_" + "2017_0101-2018_0429_less" + "_" + str(
    #         False) + ".csv"
    #     file10 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "2" + "_" + "2017_0101-2018_0429_less" + "_" + str(
    #         False) + ".csv"
    #     file11 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "3" + "_" + "2017_0101-2018_0429_less" + "_" + str(
    #         False) + ".csv"
    #     file12 = base_path_3 + "old/" + end_day + "-lightgbm_weather_params_" + "4" + "_" + "2017_0101-2018_0429_less" + "_" + str(
    #         False) + ".csv"
    #     file13 = base_path_3 + "old/" + end_day + "_weight_mean_0410.csv"
    #     file14 = base_path_3 + "old/" + end_day + "_weight_mean_0429.csv"
    #     file15 = base_path_3 + "old/" + end_day + "_weight_mean_2.csv"
    #     file16 = base_path_3 + "old/" + end_day + "_weight_mean_0410_0429.csv"
    #     ensemble_medians_with_weight([file5, file7, file8], [0.6, 0.4, 0.0], file13)
    #     ensemble_medians_with_weight([file9, file11, file12], [0.5, 0.3, 0.2], file14)
    #     ensemble_medians_with_weight([file12, file8], [0.3, 0.7], file15)
    #     ensemble_medians_with_weight([file13, file14], [0.4, 0.6], file16)
    #     score13, _, _ = test(end_day, file13)
    #     score14, _, _ = test(end_day, file14)
    #     score15, _, _ = test(end_day, file15)
    #     score16, _, _ = test(end_day, file16)
    #     now_score = np.array([score13, score14, score15, score16])
    #     print end_day, now_score
    #     score += now_score
    #     num += 1.0
    # avg_score = score / num
