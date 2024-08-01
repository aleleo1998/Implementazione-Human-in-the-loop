from flask import Flask, request, jsonify
from flask_cors import CORS
import jwt
import datetime
import time
import requests as r
import json
import os

import sys
# Aggiungi il percorso della cartella utils al percorso di ricerca dei moduli
sys.path.insert(1, '/home/aleleo/tesi_code/code_python')
from actionsUtils import getAllActions

app = Flask(__name__)
CORS(app)




#secret_key = os.urandom(24)
with open('secret_key.key', 'rb') as f:
            secret_key = f.read()

@app.route('/get-actions', methods=['GET'])
def getActions():
    return jsonify(getAllActions())



@app.route('/verify-token', methods=['POST'])
def verify_token():
    token = request.json.get('token')
    if not token:
        return jsonify({"error": "Token is missing"}), 400
    
    try:
        decoded = jwt.decode(token, secret_key, algorithms=['HS256'])
        return jsonify({"valid": True, "username": decoded['username']})
    except jwt.ExpiredSignatureError:
        return jsonify({"error": "Token has expired"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"error": "Invalid token"}), 401

def get_connetion_verifer():
    url_connection = 'http://localhost:11000/connections'
    connetcion_json = r.get(url=url_connection, json={}).json()
    return connetcion_json['results'][0]['connection_id']

def get_referent():
    url_credentials = 'http://localhost:11001/credentials'
    return r.get(url=url_credentials, json={}).json()['results'][0]['referent']

def get_pres_ex_id(thread_id, port):
    url_pres_ex_id = f"http://localhost:{port}/present-proof-2.0/records?thread_id={thread_id}"
    response = r.get(url=url_pres_ex_id)
    return response.json()['results'][0]['pres_ex_id']

def verifer_send_request(username, password):
    connection_id = get_connetion_verifer()
    url = "http://localhost:11000/present-proof-2.0/send-request"
    payload = json.dumps({
        "auto_verify": False,
        "comment": "string",
        "connection_id": connection_id,
        "presentation_request": {
            "indy": {
                "name": "Proof request",
                "non_revoked": {
                    "to": 1716243220
                },
                "nonce": "1",
                "requested_attributes": {
                    "additionalProp1": {
                        "name": "username",
                        "non_revoked": {
                            "to": 1716243220
                        },
                        "restrictions": [
                            {"attr::username::value": username}
                        ]
                    },
                    "additionalProp2": {
                        "name": "password",
                        "non_revoked": {
                            "to": 1716243220
                        },
                        "restrictions": [
                            {"attr::password::value": password}
                        ]
                    },
                    "additionalProp3": {
                        "name": "Role",
                        "non_revoked": {
                            "to": 1716243220
                        },
                        "restrictions": [
                            {"attr::Role::value": "admin"}
                        ]
                    }
                },
                "requested_predicates": {},
                "version": "1.0"
            }
        },
        "trace": False
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = r.request("POST", url, headers=headers, data=payload)
    return response.json()['thread_id']

def present_proof(thread_id):
    referent = get_referent()
    pres_ex_id = get_pres_ex_id(thread_id, 11001)
    print(referent)
    url_proof = f"http://localhost:11001/present-proof-2.0/records/{pres_ex_id}/send-presentation"
    payload_proof = json.dumps({
        "indy": {
            "requested_attributes": {
                "additionalProp1": {
                    "cred_id": str(referent),
                    "revealed": True
                },
                "additionalProp2": {
                    "cred_id": str(referent),
                    "revealed": True
                },
                "additionalProp3": {
                    "cred_id": str(referent),
                    "revealed": True
                }
            },
            "requested_predicates": {},
            "self_attested_attributes": {}
        },
        "trace": True,
        "auto_remove": False
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = r.request("POST", url_proof, headers=headers, data=payload_proof)

def verify(thread_id):
    try:
        pres_ex_id = get_pres_ex_id(thread_id, 11000)
        url_verify = f"http://localhost:11000/present-proof-2.0/records/{pres_ex_id}/verify-presentation"
        response = r.request("POST", url_verify, headers={}, data={})
        print(response.text)
        verified = response.json()['verified']
        if verified == "true":
            return True
        return False
    except:
        return False

def delete():
    url_pres_ex_id = "http://0.0.0.0:11001/present-proof-2.0/records"
    response = r.get(url=url_pres_ex_id).json()
    for el in response['results']:
        id = el['pres_ex_id']
        url_delete = f"http://0.0.0.0:11001/present-proof-2.0/records/{id}"
        r.delete(url=url_delete)

    url_pres_ex_id = "http://0.0.0.0:11000/present-proof-2.0/records"
    response = r.get(url=url_pres_ex_id).json()
    for el in response['results']:
        id = el['pres_ex_id']
        url_delete = f"http://0.0.0.0:11000/present-proof-2.0/records/{id}"
        r.delete(url=url_delete)

@app.route('/ssi-login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    delete()
    thread_id = verifer_send_request(username, password)
    time.sleep(1)
    present_proof(thread_id)
    time.sleep(1)

    response = verify(thread_id)
    if response:
        print("login effettuato")
        # Create JWT token
        token = jwt.encode({
            'username': username,
            'password': password,
            'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=2)
        }, secret_key, algorithm='HS256')
        return jsonify({"Token": token})
    else:
        print("login fallito")
        return jsonify({"error": "Login failed"}), 401

if __name__ == '__main__':
    app.run(debug=True, port=6040)
  
