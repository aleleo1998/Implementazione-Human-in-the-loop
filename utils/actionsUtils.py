from datetime import datetime
import requests as r


def addAction(user, description):
    current_date=datetime.now()
    date=str(current_date.day)+'-'+str(current_date.month)+'-'+str(current_date.year)
    url='http://127.0.0.1:5000/api/v1/namespaces/default/apis/StorageAction/invoke/storeAction'
    json={
        "input": {
            "_data": str(date),
            "_description": str(description),
            "_user": str(user)
            }
        
        }
    print(r.post(url=url, json=json))


def getAllActions():
    print('aooo')
    url='http://127.0.0.1:5000/api/v1/namespaces/default/apis/StorageAction/query/getActions'
    response=r.post(url=url,json={}).json()
    users=response['output']
    dates=response['output1']
    descriptions=response['output2']
    print(dates)
    return_list=[]
    for user, date, description in zip(users, dates, descriptions):
        return_list.append((user,date,description))

    return return_list

