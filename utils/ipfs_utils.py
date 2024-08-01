import requests as r
import encrypt as en
import os
import json
#caricamento di un file su offchain
def upload_file(filepath):
    en.crypt(filepath)
    command='curl --form autometa=true --form file=@'+filepath+' \http://localhost:5000/api/v1/namespaces/default/data'
    stream=os.popen(command)
    output=json.loads(stream.read())
    #mando in brodcast l'id a tutti così che tutti possono scaricare il file
    url1='http://127.0.0.1:5000/api/v1/namespaces/default/messages/broadcast'
    data_post={'data':[{'id':str(output['id'])}]}
    response=r.post(url1,json=data_post)
    #controllo su response
    return output['id']
    
        
def download_file(id_file):
    url='http://127.0.0.1:5000/api/v1/data/'+id_file+'/blob?limit=100'
    response=r.get(url)
    with open('result_download','w') as result_file:
        result_file.write(response.text)
    en.decrypt('result_download')
    

if __name__== '__main__':
    id=upload_file('ciao.txt')
    download_file(id)

 