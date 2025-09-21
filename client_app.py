"""flops-infra-drift: A Flower / PyTorch app."""

import time
import torch
import torchvision.models

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from task import Net, get_weights, load_data, set_weights, test, train
from collections import OrderedDict


def ShouldNodeDisconnect(partition_id, current_round):
    if (partition_id < 2):
        return False
    # For node n, partition_id is n-1
    # start_disconnect = 5, 6, 7 for partition_ids 2, 3, 4
    start_disconnect = (partition_id + 3)
    end_disconnect = 31

    return start_disconnect <= current_round < end_disconnect


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, partition_id):
        self.model = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.partition_id = partition_id

    def set_parameters(self, params):
        """Set model weights from a list of NumPy ndarrays."""
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict(
            {
                k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
                for k, v in params_dict
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        start_time = time.time()
        # Simulating client disconnection
        if (ShouldNodeDisconnect(self.partition_id, config["current_round"])):
            print("Disconnecting partition: ", self.partition_id, " for round: ", config["current_round"])
            return "Garbage"
        self.set_parameters(parameters)
        # Log cluster assignment if provided
        if "cluster_id" in config:
            print(f"[Client {self.partition_id}] Assigned cluster {config['cluster_id']}")
        train_loss = train(
            self.model,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Client: {self.partition_id} took {runtime:.4f} seconds to fit.")
        return (
            self.get_parameters({}),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        start_time = time.time()
        # Simulating client disconnection
        if (ShouldNodeDisconnect(self.partition_id, config["current_round"])):
            print("Disconnecting partition: ", self.partition_id, " for round: ", config["current_round"])
            return "Garbage"
        self.set_parameters(parameters)
        # Log cluster assignment if provided
        if "cluster_id" in config:
            print(f"[Client {self.partition_id}] Eval with cluster {config['cluster_id']}")
        loss, accuracy = test(self.model, self.valloader, self.device)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Client: {self.partition_id} took {runtime:.4f} seconds to evaluate.")
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}



def client_fn(context: Context):
    # Load model and data
    net = torchvision.models.resnet18(num_classes=10)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    # Some simulation setups provide 'local-epochs' while others may omit it.
    # Use a safe lookup with fallback to 'local_epochs' and default to 1 to
    # avoid KeyError when the key is missing.
    local_epochs_raw = context.run_config.get("local-epochs", context.run_config.get("local_epochs", 1))
    try:
        local_epochs = int(local_epochs_raw)
    except Exception:
        local_epochs = 1

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs, partition_id).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
