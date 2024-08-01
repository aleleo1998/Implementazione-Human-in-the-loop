import ipfs_utils as ipfs
import requests as r
from datetime import datetime


def addModel(filepath,name):
    id=ipfs.upload_file(filepath)
    print(id)
    current_date=datetime.now()
    date=str(current_date.day)+'-'+str(current_date.month)+'-'+str(current_date.year)
    url='http://127.0.0.1:5000/api/v1/namespaces/default/apis/ModelStorage/invoke/addModel'
    data={"input": {"_accuracy": str(0),"_data": str(date),"_id": str(id), "_name":str(name)}}
    response=r.post(url,json=data)
    print(response.text)
    return id


def getAllModel():
      url='http://127.0.0.1:5000/api/v1/namespaces/default/apis/ModelStorage/query/getAllModels'
      response=r.post(url,json={})
      print(response.text)
      names=response.json()['output']
      dates=response.json()['output1']
      ids=response.json()['output2']
      accuracies=response.json()['output3']

      return_list=[]
      for name,date,id,accuracy in zip(names, dates, ids, accuracies):
           
           return_list.append((name,date,id,accuracy))


      return return_list


    
def updateAccuracy(id,accuracy):
     accuracy= round(accuracy,2)
     url='http://127.0.0.1:5000/api/v1/namespaces/default/apis/ModelStorage/invoke/updateModelAccuracy'
     data={"input":{"_id": str(id), "_newAccuracy": str(accuracy)}}
     respone=r.post(url, json=data)



