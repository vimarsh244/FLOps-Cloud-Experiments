# MIFA (Fast Federated Learning under Device Unavailability) strategy for Flower
# Replicates the official FDU/MIFA server-side update rule in Flower's Strategy API

from logging import WARNING
from typing import Callable, Optional, Union

import numpy as np

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.server.strategy.strategy import Strategy


WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


def _zeros_like(weights: NDArrays) -> NDArrays:
    return [np.zeros_like(w, dtype=np.float32) for w in weights]


def _add_inplace(dst: NDArrays, src: NDArrays, alpha: float = 1.0) -> None:
    for i in range(len(dst)):
        dst[i] = dst[i] + np.float32(alpha) * np.asarray(src[i], dtype=np.float32)


def _copy_like(weights: NDArrays) -> NDArrays:
    return [np.asarray(w, dtype=np.float32).copy() for w in weights]


class CustomMIFA(Strategy):
    """MIFA strategy.

    Maintains a cached per-client update table U_i. On each round t, for each
    successful client i we set:

        U_i <- (w_i^t - w^t) / eta_t

    Then the global is updated using the average cached update across ALL clients:

        w^{t+1} = w^t + eta_t * (1/N) * sum_i U_i

    The server step-size follows an inverse-proportional decay schedule:
        eta_t = base_server_lr / (true_round + 1)

    This mirrors the reference FDU/MIFA implementation.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                Optional[tuple[float, dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        base_server_lr: float = 0.1,
        wait_for_all_clients_init: bool = True,
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

        # MIFA state
        self._base_lr = float(base_server_lr)
        self._true_round = 0  # counts only updates after table init
        self._table_initialized = False
        self._wait_for_all = wait_for_all_clients_init
        self._all_client_ids: set[str] = set()
        self._update_table: dict[str, NDArrays] = {}
        self._latest_global: Optional[NDArrays] = None

    def __repr__(self) -> str:
        return (
            "CustomMIFA(accept_failures="
            f"{self.accept_failures}, base_server_lr={self._base_lr})"
        )

    # Strategy lifecycle helpers
    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        initial_parameters = self.initial_parameters
        self.initial_parameters = None
        if initial_parameters is not None:
            self._latest_global = parameters_to_ndarrays(initial_parameters)
        return initial_parameters

    # Configure rounds
    def _current_eta(self) -> float:
        # As in reference: inverse-prop decay
        return self._base_lr / float(self._true_round + 1)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        # Keep global weights for this round
        if self._latest_global is None and parameters is not None:
            self._latest_global = parameters_to_ndarrays(parameters)

        # At round 1, capture the set of all initial clients known to manager
        if server_round == 1:
            try:
                all_clients_map = client_manager.all()
                self._all_client_ids = set(all_clients_map.keys())
            except Exception:
                # Fallback: leave empty, will be populated lazily from seen clients
                self._all_client_ids = set()

        # Fit config
        config: dict[str, Scalar] = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        config["current_round"] = server_round

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        fit_ins = FitIns(ndarrays_to_parameters(self._latest_global), config)
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        if self.fraction_evaluate == 0.0:
            return []
        if self._latest_global is None and parameters is not None:
            self._latest_global = parameters_to_ndarrays(parameters)
        config: dict[str, Scalar] = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        config["current_round"] = server_round
        evaluate_ins = EvaluateIns(ndarrays_to_parameters(self._latest_global), config)

        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        return [(client, evaluate_ins) for client in clients]

    # Aggregation
    def _ensure_table_entries(self) -> None:
        # Initialize zeros for any clients discovered but missing from table
        for cid in self._all_client_ids:
            if cid not in self._update_table and self._latest_global is not None:
                self._update_table[cid] = _zeros_like(self._latest_global)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        if self._latest_global is None:
            # Should not happen; fallback to any result's params as starting point
            self._latest_global = parameters_to_ndarrays(results[0][1].parameters)

        # Expand known clients set with any newly seen ids
        for client, _ in results:
            cid = getattr(client, "cid", None) or str(client)
            self._all_client_ids.add(cid)

        self._ensure_table_entries()

        # Compute normalized updates and update the table for responding clients
        eta_t = self._current_eta()
        old_global = _copy_like(self._latest_global)
        for client, fit_res in results:
            cid = getattr(client, "cid", None) or str(client)
            w_i = parameters_to_ndarrays(fit_res.parameters)
            # delta = w_i - w_t
            upd = [np.asarray(w_i[k], dtype=np.float32) - np.asarray(old_global[k], dtype=np.float32) for k in range(len(w_i))]
            # normalized update: (w_i - w_t)/eta_t
            norm_upd = [u / np.float32(eta_t) for u in upd]
            self._update_table[cid] = norm_upd

        # If waiting for all clients, only start true rounds after all have at least one entry
        if self._wait_for_all:
            # Initialize table for any clients not yet seen
            self._ensure_table_entries()
            # Check if every known client has at least one update (non-None)
            if not self._table_initialized:
                have_all = len(self._update_table) > 0 and all(
                    cid in self._update_table for cid in self._all_client_ids
                ) and len(self._all_client_ids) > 0
                if not have_all:
                    # Keep global unchanged until initialized; return current params
                    params = ndarrays_to_parameters(self._latest_global)
                    # Aggregate custom metrics if provided
                    metrics_aggregated: dict[str, Scalar] = {}
                    if self.fit_metrics_aggregation_fn:
                        fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                        metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
                    elif server_round == 1:
                        log(WARNING, "No fit_metrics_aggregation_fn provided")
                    return params, metrics_aggregated
                # Mark table initialized now
                self._table_initialized = True

        # Compute mean of cached updates across all known clients
        self._ensure_table_entries()
        N = max(1, len(self._all_client_ids))
        mean_upd = _zeros_like(self._latest_global)
        for cid in self._all_client_ids:
            _add_inplace(mean_upd, self._update_table[cid], alpha=1.0 / float(N))

        # Apply server update: w_{t+1} = w_t + eta_t * mean(U)
        new_global = _copy_like(old_global)
        _add_inplace(new_global, mean_upd, alpha=eta_t)
        self._latest_global = new_global
        self._true_round += 1

        params = ndarrays_to_parameters(self._latest_global)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return params, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        if self.evaluate_fn is None:
            return None
        params_nd = self._latest_global
        if params_nd is None and parameters is not None:
            params_nd = parameters_to_ndarrays(parameters)
        if params_nd is None:
            return None
        loss, metrics = self.evaluate_fn(server_round, params_nd, {}) or (None, None)
        if loss is None:
            return None
        return loss, metrics or {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated: dict[str, Scalar] = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")
        return loss_aggregated, metrics_aggregated
