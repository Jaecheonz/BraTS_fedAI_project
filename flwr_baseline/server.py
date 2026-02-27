from __future__ import annotations

import flwr as fl
import numpy as np

def weighted_average(metrics):
    num = {}
    den = {}
    for n, m in metrics:
        for k, v in m.items():
            v = float(v)
            if np.isnan(v):
                continue
            num[k] = num.get(k, 0.0) + n * v
            den[k] = den.get(k, 0.0) + n
    return {k: num[k] / den[k] for k in num.keys() if den.get(k, 0) > 0}


def make_strategy(num_clients: int, local_epochs: int = 1):
    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=lambda rnd: {"local_epochs": local_epochs},
        evaluate_metrics_aggregation_fn=weighted_average,  # <-- key line
    )