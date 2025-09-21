"""Run Flower Simulation for this app (server_app + client_app)."""

from flwr.simulation import run_simulation

# Import the App objects
from server_app import app as server_app
from client_app import app as client_app


if __name__ == "__main__":
    # Configure simulation
    NUM_SUPERNODES = 10  # number of simulated clients

    # Optional: specify resources per client (works with Ray backend)
    backend_config = {
        "init_args": {"num_cpus": 4},
        "client_resources": {"num_cpus": 1},
    }

    # Run
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=NUM_SUPERNODES,
        backend_name="ray",
        backend_config=backend_config,
        verbose_logging=False,
    )
