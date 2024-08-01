import argparse
from typing import List
import warnings
from collections import OrderedDict
from flwr.client import NumPyClient, ClientApp
from flwr_datasets import FederatedDataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import torchvision.transforms as transforms
from flask import Flask, jsonify
app1= Flask(__name__)
from flask_socketio import SocketIO, emit
socketio = SocketIO(app1, cors_allowed_origins="*")  # Inizializza SocketIO



# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--partition-id",
    choices=[0, 1, 2,3,4,5,6,7,8,9],
    default=0,
    type=int,
    help="Partition of the dataset divided into 10 iid partitions created artificially.",
)
parser.add_argument(
    "--port",
    default=0,
    type=int,
)

partition_id = parser.parse_known_args()[0].partition_id
port = parser.parse_known_args()[0].port
print(port)
print(partition_id)

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


"""class VGG(nn.Module):
    def __init__(self, vgg_name="VGG16"):
      super(VGG, self).__init__()
      self.features = self._make_layers(cfg[vgg_name])
      self.classifier = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Linear(512, 10)
      )

    def forward(self, x):
      out = self.features(x)
      out = out.view(out.size(0), -1)
      out = self.classifier(out)
      output = F.log_softmax(out, dim=1)
      return output

    def _make_layers(self, cfg):
      layers = []
      in_channels = 3
      for x in cfg:
        if x == 'M':
          layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
          layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                     nn.BatchNorm2d(x),
                     nn.ReLU(inplace=True)]
          in_channels = x
      layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
      return nn.Sequential(*layers)

"""
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



def train(net, trainloader, epochs: int):
    """Train the network on the training set."""
    # Lista per memorizzare i valori di epoch_loss e accuracy 
  
    socketio.emit('startTraining')
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()
    for epoch in range(epochs):
        
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in tqdm(trainloader,'Training'):
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Client {partition_id}Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
        socketio.emit('data',f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
    
    socketio.emit('stopTraining')
        # Stampa e memorizza i valori di loss e accuracy
        

def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in tqdm(testloader, "Testing"):
            images = batch["img"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def load_data(partition_id):
    """Load partition CIFAR10 data."""
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": 10})
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms =Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader




# Load model and data (simple CNN, CIFAR-10)
net =Net().to(DEVICE)
trainloader, testloader = load_data(partition_id=partition_id)


# Define Flower client
class FlowerClient(NumPyClient):

    def __init__(self,cid) -> None:
        super().__init__()
        self.cid=cid
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

   
    def set_parameters(self, parameters: List[np.ndarray]):
        new_state_dict = OrderedDict()
        model_state_dict = net.state_dict()

        for key, param in zip(model_state_dict.keys(), parameters):
            if param.shape != model_state_dict[key].shape:
                raise ValueError(f"Shape mismatch for {key}: expected {model_state_dict[key].shape}, got {param.shape}")
            new_state_dict[key] = torch.from_numpy(param).to(DEVICE)

        net.load_state_dict(new_state_dict, strict=True)

        
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=2)
        return self.get_parameters(config={}), len(trainloader.dataset), {'cid':self.cid}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    return FlowerClient().to_client()


# Flower ClientApp
appClient = ClientApp(
    client_fn=client_fn,
)

@socketio.on('startsimulation')
def startclient():
    print("ciao")
    from flwr.client import start_client
    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(str(partition_id)).to_client(),
    )



# Legacy mode
if __name__ == "__main__":
    socketio.run(app1,port=int(port))
    #startclient()
    #train(net,trainloader,100)
    #print(test(net,testloader))
