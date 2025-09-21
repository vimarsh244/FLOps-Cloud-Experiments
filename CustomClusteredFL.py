# Clustered Federated Learning strategy for Flower
# Based on Sattler et al. 2019, adapted to Flower Strategy API

from logging import WARNING
from typing import Callable, Optional, Union

import numpy as np
from collections import defaultdict

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


class CustomClusteredFL(Strategy):
    """Clustered Federated Learning strategy (Sattler et al.).

    Maintains per-cluster models and sends each client the model of its cluster.
    Supports dynamic client disconnect/rejoin and on-the-fly cluster splits.
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
            Callable[[int, NDArrays, dict[str, Scalar]], Optional[tuple[float, dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
        # CFL specific knobs
        eps1_mean_update_norm: float = 0.05,
        eps2_max_update_norm: float = 0.2,
        split_warmup_rounds: int = 5,
        split_cooldown_rounds: int = 3,
        min_clients_for_split: int = 3,
        min_cluster_size: int = 2,
        max_kmeans_iters: int = 10,
    ) -> None:
        super().__init__()
        # General
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
        self.inplace = inplace

        # CFL state
        self._cluster_models: dict[int, NDArrays] = {}
        self._client_to_cluster: dict[str, int] = {}
        self._cluster_clients: dict[int, set[str]] = defaultdict(set)
        self._round_assignments: dict[int, dict[str, int]] = {}
        self._last_split_round: int = 0
        self._next_cluster_id: int = 0

        # CFL params
        self.eps1 = eps1_mean_update_norm
        self.eps2 = eps2_max_update_norm
        self.split_warmup_rounds = split_warmup_rounds
        self.split_cooldown_rounds = split_cooldown_rounds
        self.min_clients_for_split = min_clients_for_split
        self.min_cluster_size = min_cluster_size
        self.max_kmeans_iters = max_kmeans_iters

    def __repr__(self) -> str:
        return (
            "CustomClusteredFL(accept_failures="
            f"{self.accept_failures}, clusters={len(self._cluster_models)})"
        )

    # Strategy lifecycle     def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize the first cluster with the provided initial parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None
        if initial_parameters is not None:
            self._cluster_models = {0: parameters_to_ndarrays(initial_parameters)}
            self._next_cluster_id = 1
        return initial_parameters

    def _select_initial_cluster_for(self, cid: str) -> int:
        # If client seen before, reuse assignment; else use largest cluster or 0
        if cid in self._client_to_cluster:
            return self._client_to_cluster[cid]
        if not self._cluster_models:
            # Will be lazily initialized in configure_* from provided server parameters
            return 0
        # Choose the cluster with most members
        if self._cluster_clients:
            sizes = {k: len(v) for k, v in self._cluster_clients.items()}
            return max(sizes, key=sizes.get)
        return 0

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        # Fit config
        config: dict[str, Scalar] = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        config["current_round"] = server_round

        # Seed default cluster model if uninitialized
        if not self._cluster_models and parameters is not None:
            self._cluster_models[0] = parameters_to_ndarrays(parameters)
            self._next_cluster_id = 1

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Assign per-client cluster and send that cluster's model
        round_assignments: dict[str, int] = {}
        fit_instructions: list[tuple[ClientProxy, FitIns]] = []
        for client in clients:
            cid = getattr(client, "cid", None) or str(client)
            cluster_id = self._select_initial_cluster_for(cid)
            round_assignments[cid] = cluster_id
            self._client_to_cluster[cid] = cluster_id
            self._cluster_clients[cluster_id].add(cid)
            cluster_params = ndarrays_to_parameters(self._cluster_models[cluster_id])
            cfg = {**config, "cluster_id": cluster_id}
            fit_instructions.append((client, FitIns(cluster_params, cfg)))

        # Save assignments for this round to use during aggregation
        self._round_assignments[server_round] = round_assignments
        return fit_instructions

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        if self.fraction_evaluate == 0.0:
            return []
        config: dict[str, Scalar] = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        config["current_round"] = server_round

        # Seed default cluster model if uninitialized
        if not self._cluster_models and parameters is not None:
            self._cluster_models[0] = parameters_to_ndarrays(parameters)
            self._next_cluster_id = 1

        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        eval_instructions: list[tuple[ClientProxy, EvaluateIns]] = []
        for client in clients:
            cid = getattr(client, "cid", None) or str(client)
            cluster_id = self._client_to_cluster.get(cid, self._select_initial_cluster_for(cid))
            params = ndarrays_to_parameters(self._cluster_models[cluster_id])
            cfg = {**config, "cluster_id": cluster_id}
            eval_instructions.append((client, EvaluateIns(params, cfg)))
        return eval_instructions

    # Aggregation helpers
    @staticmethod
    def _weighted_aggregate(results: list[tuple[NDArrays, int]], inplace: bool) -> NDArrays:
        if not results:
            raise ValueError("No results to aggregate")
        # Determine dtype behavior per-parameter
        first_weights = results[0][0]
        is_float = [np.issubdtype(np.asarray(w).dtype, np.floating) for w in first_weights]
        if inplace:
            base = [
                (np.asarray(w, dtype=np.float32).copy() if is_float[i] else np.asarray(w).copy())
                for i, w in enumerate(first_weights)
            ]
            total = results[0][1]
            for arrs, n in results[1:]:
                denom = total + n
                alpha = np.float32(n / denom) if denom > 0 else np.float32(0.0)
                for i, arr in enumerate(arrs):
                    if is_float[i]:
                        arr_f = np.asarray(arr, dtype=np.float32)
                        base[i] += alpha * (arr_f - base[i])
                    else:
                        # Keep the original non-float parameter (no averaging)
                        pass
                total += n
            return base
        # Copy-based weighted average
        total_examples = sum(n for _, n in results)
        if total_examples == 0:
            return [np.asarray(w, dtype=np.float32) if is_float[i] else np.asarray(w).copy() for i, w in enumerate(first_weights)]
        agg: NDArrays = []
        # Initialize accumulators
        for i, w in enumerate(first_weights):
            if is_float[i]:
                agg.append(np.zeros_like(np.asarray(w), dtype=np.float32))
            else:
                agg.append(np.asarray(w).copy())  # placeholder; will keep first
        # Accumulate
        for weights, n in results:
            w_factor = np.float32(n / total_examples)
            for i, w in enumerate(weights):
                if is_float[i]:
                    agg[i] += w_factor * np.asarray(w, dtype=np.float32)
                else:
                    # Leave as the first occurrence
                    pass
        return agg

    @staticmethod
    def _flatten_params_difference(new: NDArrays, old: NDArrays) -> np.ndarray:
        vecs = [(np.asarray(n, dtype=np.float32) - np.asarray(o, dtype=np.float32)).ravel() for n, o in zip(new, old)]
        if not vecs:
            return np.array([], dtype=np.float32)
        return np.concatenate([v for v in vecs])

    def _binary_spherical_kmeans(self, X: np.ndarray) -> np.ndarray:
        """Very small k=2 spherical k-means for splitting. Returns labels in {0,1}."""
        m = X.shape[0]
        if m < 2:
            return np.zeros(m, dtype=int)
        # Normalize
        eps = 1e-12
        norms = np.linalg.norm(X, axis=1, keepdims=True) + eps
        Xn = X / norms
        # Init centers by picking two farthest points (by cosine distance)
        S = Xn @ Xn.T
        i = 0
        j = int(np.argmin(S[0]))
        for _ in range(2):
            i = int(np.argmin(S[j]))
            j = int(np.argmin(S[i]))
        c0, c1 = Xn[i], Xn[j]
        labels = np.zeros(m, dtype=int)
        for _ in range(self.max_kmeans_iters):
            # Assign
            sim0 = Xn @ c0
            sim1 = Xn @ c1
            new_labels = (sim1 > sim0).astype(int)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            # Update centers
            if np.any(labels == 0):
                c0 = Xn[labels == 0].mean(axis=0)
                c0_norm = np.linalg.norm(c0) + eps
                c0 = c0 / c0_norm
            if np.any(labels == 1):
                c1 = Xn[labels == 1].mean(axis=0)
                c1_norm = np.linalg.norm(c1) + eps
                c1 = c1 / c1_norm
        return labels

    # Aggregate & split
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

        # Group results by cluster used for training in this round
        round_assign = self._round_assignments.get(server_round, {})
        grouped: dict[int, list[tuple[str, NDArrays, int]]] = defaultdict(list)
        for client, fit_res in results:
            cid = getattr(client, "cid", None) or str(client)
            params_nd = parameters_to_ndarrays(fit_res.parameters)
            n = fit_res.num_examples
            cluster_id = round_assign.get(cid, self._client_to_cluster.get(cid, 0))
            grouped[cluster_id].append((cid, params_nd, n))

        # For stats and splitting: collect per-cluster update vectors vs old cluster model
        cluster_new_models: dict[int, NDArrays] = {}
        split_candidates: list[int] = []
        for cid_cluster, items in grouped.items():
            old_model = self._cluster_models[cid_cluster]
            # Weighted aggregate to form updated cluster model
            agg = self._weighted_aggregate([(p, n) for _, p, n in items], self.inplace)
            cluster_new_models[cid_cluster] = agg

            # Compute update vectors
            upd_vecs = []
            for _, p, _ in items:
                dv = self._flatten_params_difference(p, old_model)
                upd_vecs.append(dv)
            if len(upd_vecs) >= self.min_clients_for_split:
                norms = np.array([np.linalg.norm(v) for v in upd_vecs])
                mean_norm = float(np.mean(norms))
                max_norm = float(np.max(norms))
                if (
                    server_round >= self.split_warmup_rounds
                    and server_round - self._last_split_round >= self.split_cooldown_rounds
                    and mean_norm < self.eps1
                    and max_norm > self.eps2
                ):
                    split_candidates.append(cid_cluster)
                log(WARNING, f"[CFL] Round {server_round} cluster {cid_cluster}: mean_norm={mean_norm:.4f}, max_norm={max_norm:.4f}")

        # Apply splits (at most one split per round to keep things simple)
        for cid_cluster in split_candidates[:1]:
            items = grouped[cid_cluster]
            old_model = self._cluster_models[cid_cluster]
            X = np.stack([
                self._flatten_params_difference(p, old_model) for _, p, _ in items
            ], axis=0)
            labels = self._binary_spherical_kmeans(X)
            g0 = [it for it, lbl in zip(items, labels) if lbl == 0]
            g1 = [it for it, lbl in zip(items, labels) if lbl == 1]
            if len(g0) >= self.min_cluster_size and len(g1) >= self.min_cluster_size:
                # Create/update two clusters
                # Reuse original id for larger group for stability
                if len(g1) > len(g0):
                    g0, g1 = g1, g0  # swap so g0 is the larger group
                # Aggregate group models
                new0 = self._weighted_aggregate([(p, n) for _, p, n in g0], self.inplace)
                new1 = self._weighted_aggregate([(p, n) for _, p, n in g1], self.inplace)

                # Update original cluster with group 0, create new id for group 1
                self._cluster_models[cid_cluster] = new0
                new_cluster_id = self._next_cluster_id
                self._next_cluster_id += 1
                self._cluster_models[new_cluster_id] = new1

                # Reassign clients from this round
                self._cluster_clients[cid_cluster].clear()
                self._cluster_clients[new_cluster_id].clear()
                for (cid, _, _), lbl in zip(items, labels):
                    assigned = cid_cluster if (lbl == 0) else new_cluster_id
                    self._client_to_cluster[cid] = assigned
                    self._cluster_clients[assigned].add(cid)

                self._last_split_round = server_round
                log(WARNING, f"[CFL] Split cluster {cid_cluster} -> {cid_cluster} & {new_cluster_id} at round {server_round}")
            else:
                log(WARNING, f"[CFL] Skip split cluster {cid_cluster}: small group sizes ({len(g0)}, {len(g1)})")

        # For clusters without split, update to new aggregated model
        for c, new_model in cluster_new_models.items():
            self._cluster_models[c] = new_model

        # Build a global placeholder parameters (weighted avg over clusters by participating examples)
        total_examples = sum(n for items in grouped.values() for *_, n in items)
        if total_examples == 0:
            # Fallback: return any cluster model
            any_params = next(iter(self._cluster_models.values()))
            global_params = ndarrays_to_parameters(any_params)
        else:
            # Weighted by examples counted in this round (clusters without updates will be ignored)
            accum = None
            for c, items in grouped.items():
                n_c = sum(n for *_, n in items)
                w = np.float32(n_c / total_examples)
                model = self._cluster_models[c]
                if accum is None:
                    accum = [np.asarray(ww, dtype=np.float32) * w for ww in model]
                else:
                    for i in range(len(accum)):
                        accum[i] += w * np.asarray(model[i], dtype=np.float32)
            global_params = ndarrays_to_parameters(accum if accum is not None else next(iter(self._cluster_models.values())))

        # Aggregate custom metrics if a function provided
        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return global_params, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        if self.evaluate_fn is None:
            return None
        # Use a representative model (largest cluster) for centralized eval
        if self._cluster_clients:
            largest = max(self._cluster_clients.items(), key=lambda kv: len(kv[1]))[0]
            params_nd = self._cluster_models[largest]
        else:
            # Fallback: any cluster model
            params_nd = next(iter(self._cluster_models.values()))
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
            [(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results]
        )
        metrics_aggregated: dict[str, Scalar] = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")
        return loss_aggregated, metrics_aggregated
