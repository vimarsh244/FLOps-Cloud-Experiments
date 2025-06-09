import time
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Callable

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from flwr.common import (
    Context,
    Metrics,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    Scalar,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerConfig
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAdam, FedAdagrad
from flwr.client import ClientApp, NumPyClient
from flwr.simulation import start_simulation
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

# #############################################################################
# 1. CONFIGURATION
# #############################################################################

# Set the desired strategy: "FedAdam" or "FedAdagrad"
STRATEGY = "FedAdagrad"

# Set data distribution: True for IID, False for non-IID
IID_DISTRIBUTION = False

# Simulation parameters
NUM_CLIENTS = 5
NUM_ROUNDS = 50
LOCAL_EPOCHS = 3
FRACTION_FIT = 1.0


# #############################################################################
# 2. DATASET AND MODEL DEFINITION (from task.py)
# #############################################################################

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset

def load_data(partition_id: int, num_partitions: int, iid: bool):
    """Load partition of CIFAR-10 data."""
    global fds
    if fds is None:
        if iid:
            partitioner = IidPartitioner(num_partitions=num_partitions)
        else:
            # Alpha controls the level of non-IID-ness, smaller alpha means more non-IID
            partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="label", alpha=0.5)
        
        fds = FederatedDataset(
            dataset="cifar10",
            partitioners={"train": partitioner},
        )
    
    partition = fds.load_partition(partition_id, "train")
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    avg_loss = loss / len(testloader)
    return avg_loss, accuracy


# #############################################################################
# 3. CLIENT DEFINITION (from client_app.py)
# #############################################################################

def ShouldNodeDisconnect(partition_id, current_round):
    """Determines if a client should drop out based on its ID and the current round."""
    if partition_id < 2:
        return False
    # For node n, partition_id is n-1
    # start_disconnect = 5, 6, 7 for partition_ids 2, 3, 4
    start_disconnect = partition_id + 3
    end_disconnect = 31  # Kept high to ensure they stay disconnected

    return start_disconnect <= current_round < end_disconnect


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, partition_id):
        self.model = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.partition_id = partition_id

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        start_time = time.time()
        # Simulate client disconnection by raising an exception
        if ShouldNodeDisconnect(self.partition_id, config["current_round"]):
            print(f"Disconnecting Client {self.partition_id} for round {config['current_round']}")
            raise Exception(f"Simulating client {self.partition_id} dropout.")
        
        self.set_parameters(parameters)
        train_loss = train(self.model, self.trainloader, self.local_epochs, self.device)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Client {self.partition_id} took {runtime:.4f} seconds to fit.")
        
        return self.get_parameters({}), len(self.trainloader.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        start_time = time.time()
        # Simulate client disconnection
        if ShouldNodeDisconnect(self.partition_id, config["current_round"]):
            print(f"Disconnecting Client {self.partition_id} for round {config['current_round']}")
            raise Exception(f"Simulating client {self.partition_id} dropout.")

        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Client {self.partition_id} took {runtime:.4f} seconds to evaluate.")
        
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}


def client_fn(cid: str):
    """Create a Flower client instance."""
    net = Net()
    partition_id = int(cid)
    trainloader, valloader = load_data(partition_id, NUM_CLIENTS, IID_DISTRIBUTION)
    return FlowerClient(net, trainloader, valloader, LOCAL_EPOCHS, partition_id)


# #############################################################################
# 4. SERVER DEFINITION (from server_app.py)
# #############################################################################

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics for federated evaluation."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics if "accuracy" in m]
    examples = [num_examples for num_examples, m in metrics if "accuracy" in m]

    if not examples:
        return {"accuracy": 0}
        
    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return training configuration dict for each round."""
    config = {
        "current_round": server_round,
    }
    return config

def get_strategy(strategy_name: str, initial_parameters: Parameters):
    """Return the specified strategy."""
    if strategy_name.lower() == "fedadam":
        return FedAdam(
            initial_parameters=initial_parameters,
            fraction_fit=FRACTION_FIT,
            fraction_evaluate=1.0,
            min_fit_clients=int(NUM_CLIENTS * FRACTION_FIT),
            min_available_clients=NUM_CLIENTS,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=fit_config,  # <-- BUG FIX: Pass config to evaluate
            evaluate_metrics_aggregation_fn=weighted_average,
            # FedAdam parameters
            eta=0.01,
            beta_1=0.9,
            beta_2=0.999,
            tau=1e-9,
        )
    elif strategy_name.lower() == "fedadagrad":
        return FedAdagrad(
            initial_parameters=initial_parameters,
            fraction_fit=FRACTION_FIT,
            fraction_evaluate=1.0,
            min_fit_clients=int(NUM_CLIENTS * FRACTION_FIT),
            min_available_clients=NUM_CLIENTS,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=fit_config,  # <-- BUG FIX: Pass config to evaluate
            evaluate_metrics_aggregation_fn=weighted_average,
            # FedAdagrad parameters
            eta=0.01,
            tau=1e-9,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


# #############################################################################
# 5. SIMULATION EXECUTION
# #############################################################################

if __name__ == "__main__":
    print(f"Running simulation with strategy: {STRATEGY}")
    print(f"Data distribution is {'IID' if IID_DISTRIBUTION else 'Non-IID'}")
    
    # Create an initial model and get its parameters
    initial_net = Net()
    initial_params = ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in initial_net.state_dict().items()]
    )

    # Define strategy
    strategy = get_strategy(STRATEGY, initial_params)

    # Start simulation
    history = start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0.0},
    )

    # Print results
    print("=" * 80)
    print(f"SIMULATION RESULTS - STRATEGY: {STRATEGY}")
    print("=" * 80)
    print(f"History (loss, distributed): {history.losses_distributed}")
    print(f"History (metrics, distributed): {history.metrics_distributed}")

    # save the history to a file
    import json
    with open("simulation_history_fedadam_non_iid.json", "w") as f:
        json.dump({
            "losses_distributed": history.losses_distributed,
            "metrics_distributed": history.metrics_distributed,
        }, f, indent=4)


    # PLOTTING results
    # Prepare loss data
    rounds_loss = [r for r, _ in history.losses_distributed]
    losses = [l for _, l in history.losses_distributed]

    # Prepare accuracy data
    acc_tuples = history.metrics_distributed.get("accuracy", [])
    rounds_acc = [r for r, _ in acc_tuples]
    accuracies = [a for _, a in acc_tuples]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rounds_loss, losses, marker="o", label="Loss (distributed)")
    plt.title("Loss Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(rounds_acc, accuracies, marker="o", label="Accuracy (distributed)")
    plt.title("Accuracy Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
