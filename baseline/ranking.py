# -*- coding:utf-8 -*-
import requests
import json
import numpy as np


def cal_rank(move_k=7):
    link = "https://biendata.com/competition/kdd_2018_leaderboard_data/"
    respones = requests.get(link)
    data = json.loads(respones.text)
    ans = {}
    for i in range(1, len(data)):

        team_name = data[i]["team_name"]
        ans[team_name] = {}
        tmp = []
        for k in data[i].keys():
            if k[0] == "A":
                # print k
                tmp.append(float(data[i][k]))
        ans[team_name]["score"] = np.sort(np.array(tmp))
        # print ans[team_name]["score"]
        if move_k == 0:
            ans[team_name]["avg_score"] = np.mean(ans[team_name]["score"][:])
        else:
            ans[team_name]["avg_score"] = np.mean(ans[team_name]["score"][:-move_k])
    ans = sorted(ans.items(), lambda x, y: cmp(x[1]["avg_score"], y[1]["avg_score"]))
    for i in range(30):
        print i+1, ans[i][0], ans[i][1]['avg_score'], ans[i][1]['score']


if __name__ == '__main__':
    cal_rank(move_k=6)
