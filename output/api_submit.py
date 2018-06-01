
# coding: utf-8

import requests
base_path = '../image/results/20180531/'
files={'files': open(base_path +'2018-05-31_ensemble_all_zhoujie.csv','rb')}

# data = {
#     "user_id": "zhoujie",   # user_id is your username which can be found on the top-right corner on our website when you logged in.
#     "team_token": "c4bba7c520632e0e623e7db559e434ee12d61dfb2b594fa55694733390bba651", #your team_token.
#     "description": 'ensemble',  # no more than 40 chars.
#     "filename": "2018-05-03lightgbm_median_ensemble", # your filename
# }
#
data = {
    "user_id": "poteman",   #user_id is your username which can be found on the top-right corner on our website when you logged in.
    "team_token": "925a4b65ba78840b764f6d413120e4d93b982974107721ee3ddd45c1cc5b92dc", #your team_token.
    "description": 'ensemble',  #no more than 40 chars.
    "filename": "2018-05-31_ensemble_all_zhoujie", #your filename
}

url = 'https://biendata.com/competition/kdd_2018_submit/'

response = requests.post(url, files=files, data=data)

print(response.text)

