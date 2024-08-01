import asyncio
from typing import Any, Callable, Union
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import flwr as fl
import numpy as np
import sys
import torch.nn as nn
import jwt
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../utils')

sys.path.insert(2, '../login_ssi')




import actionsUtils as actionsUtils
import ipfs_utils
from flwr.common import FitRes, MetricsAggregationFn

from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,

)
from flask import Flask, jsonify

import os
import torch.nn.functional as F
import pickle
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
import torch
import model_utils

app = Flask(__name__)
from flask_socketio import SocketIO, emit
socketio = SocketIO(app, cors_allowed_origins="*") 


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):

    def __init__(self, *, fraction_fit: float = 1, fraction_evaluate: float = 1, min_fit_clients: int = 2, 
                 min_evaluate_clients: int = 2, min_available_clients: int = 2, 
                 evaluate_fn: Callable[[int, List[np.ndarray[Any, np.dtype[Any]]], Dict[str, bool | bytes | float | int | str]], Tuple[float, Dict[str, bool | bytes | float | int | str]] | None] | None = None, 
                 on_fit_config_fn: Callable[[int], Dict[str, bool | bytes | float | int | str]] | None = None, 
                 on_evaluate_config_fn: Callable[[int], Dict[str, bool | bytes | float | int | str]] | None = None, accept_failures: bool = True, initial_parameters: Parameters | None = None, fit_metrics_aggregation_fn: Callable[[List[Tuple[int, Dict[str, bool | bytes | float | int | str]]]], Dict[str, bool | bytes | float | int | str]] | None = None, evaluate_metrics_aggregation_fn: Callable[[List[Tuple[int, Dict[str, bool | bytes | float | int | str]]]], Dict[str, bool | bytes | float | int | str]] | None = None, inplace: bool = True) -> None:
        
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, 
                         min_fit_clients=min_fit_clients, 
                         min_evaluate_clients=min_evaluate_clients, 
                         min_available_clients=min_available_clients, 
                         evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, 
                         on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, 
                         initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, 
                         evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)
        self.id=""
        self.name_file=""
        self.username=""
        socketio.on_event('setUsername', self.setUsername)



    
    
    def setUsername(self,token):
        with open('secret_key.key', 'rb') as f:
            secret_key = f.read()

        decoded = jwt.decode(token, secret_key, algorithms=['HS256'])


        self.username=decoded['username']
       
       
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation accuracy using weighted average."""

        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics

      
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}, {self.id}")
        socketio.emit('serverResults',f"Round {server_round} aggregated accuracy  from client results: {aggregated_accuracy}")
        model_utils.updateAccuracy(self.id, aggregated_accuracy)
        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}
    
   
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
      
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        
        
        cids=[]
        for r,s in results:
            cids.append(s.metrics['cid'])
    
        
        socketio.emit('startAggragate')
        socketio.emit('AddClientServer',cids)
        
        s=""
        
        sblocco=False
        
        @socketio.on('aggregate_fit_input')
        def function_input(message):
            nonlocal sblocco
            nonlocal s
            print(type(message))
            sblocco=True
            s=message 

        while sblocco==False:
            pass
            
  

        
        if s!=[]:
            for exclude in s:
                index=-1
                exclude=int(exclude)


                for i, cid in enumerate(cids):
                
                    if int(cid)==exclude:
                        print('trovato')
                        actionsUtils.addAction(self.username, f"ha escluso dall'aggreagazione il client {exclude+1} durante il round {server_round}")
                        index=i
                        break
                    

                if index!=-1:
                    results.pop(index)
                    cids.pop(index)


                

        
        socketio.emit('endAggregate')
        
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
 
        net=Net().to('cpu')
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        list_models=model_utils.getAllModel()
        for el in list_models:
            print(el)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict,strict=True)
            torch.save(net.state_dict(), f"model_rounds/model_round_{server_round}.pth")
            self.name=f'model_round_{server_round}'
            self.id=model_utils.addModel(f'model_rounds/model_round_{server_round}.pth',self.name)
          
            
        

        

        socketio.emit('downloadModel',list_models)

        sblocco=False
        downloadFile=""
        @socketio.on('downloadModel')
        def function_donwload(message):
            nonlocal sblocco
            nonlocal downloadFile
            downloadFile=message 
            sblocco=True
            

        while sblocco==False:
            pass

        
        socketio.emit('endDownloadModel')
       
        if downloadFile!='':
           ipfs_utils.download_file(downloadFile)
           actionsUtils.addAction(self.username, f'ha caricato il modello {downloadFile} durante il round {server_round} ')
           net.load_state_dict(torch.load('result_download'))
           state_dict_ndarrays = [v.cpu().numpy() for v in net.state_dict().values()]
           parametri=fl.common.ndarrays_to_parameters(state_dict_ndarrays)
          
           return parametri, aggregated_metrics
           
        
        return  aggregated_parameters, aggregated_metrics


strategy = AggregateCustomMetricStrategy(
        fraction_fit=0.9,
        fraction_evaluate=0.5,
        min_evaluate_clients=1,
        min_available_clients=4,
       
    )

from flwr.server import ServerApp, ServerConfig
Serverapp = ServerApp(
    config={},
    strategy=strategy,
)
@socketio.on('startserver')
def start_server():
    # Start Flower server
  fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=50),
    strategy=strategy
  )
  
if __name__=='__main__':
    #start_server()
    socketio.run(app,port=6020)
