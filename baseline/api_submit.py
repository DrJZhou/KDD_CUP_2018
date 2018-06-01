# coding: utf-8

import requests
from datetime import datetime, timedelta
base_path = "../output/"


def submit(filename):

    files = {'files': open(base_path + filename, 'rb')}

    data = {
        "user_id": "poteman",
    # user_id is your username which can be found on the top-right corner on our website when you logged in.
        "team_token": "925a4b65ba78840b764f6d413120e4d93b982974107721ee3ddd45c1cc5b92dc",  # your team_token.
        "description": 'ensemble',  # no more than 40 chars.
        "filename": filename,  # your filename
    }

    url = 'https://biendata.com/competition/kdd_2018_submit/'
    response = requests.post(url, files=files, data=data)
    print(response.text)


if __name__ == '__main__':
    time_now = datetime.now()
    time_now = time_now - timedelta(hours=8)
    start_day = (time_now - timedelta(days=2)).strftime('%Y-%m-%d')
    end_day = time_now.strftime('%Y-%m-%d')
    print end_day
    end_day = "2018-05-07"
    filename = end_day + "_ensemble_all_zhoujie.csv"
    submit(filename=filename)
