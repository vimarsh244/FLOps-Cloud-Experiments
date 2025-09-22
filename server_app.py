"""flops-infra-drift: A Flower / PyTorch app."""


from CustomClusteredFL import CustomClusteredFL
# from CustomFedProx import CustomFedProx  # optional
from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
# from flwr.server.strategy import FedAvg, FedProx
from typing import List, Tuple
from flwr.common import Metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    total_examples = sum(examples) if examples else 1
    return {"accuracy": (sum(accuracies) / total_examples) if total_examples else 0.0}


def server_fn(context: Context):
    # Read from config with defaults
    num_rounds = context.run_config.get("num-server-rounds", 10)
    fraction_fit = context.run_config.get("fraction-fit", 1.0)

    # Define strategy (Clustered FL)
    strategy = CustomClusteredFL(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_available_clients=5,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)