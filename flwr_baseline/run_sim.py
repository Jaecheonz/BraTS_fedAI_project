# run_sim.py
# python -m flwr_baseline.run_sim
from __future__ import annotations
from pathlib import Path
import torch
import csv
from torch.utils.data import DataLoader
import flwr as fl
from .dataset import list_cases, list_cases_raw, partition_cases, BratsPatchDataset, BratsCaseDataset
from .client import BratsClient
from .server import make_strategy
from datetime import datetime
import logging
logging.getLogger("flwr").propagate = False

def client_fn(cid: str, pre_root: Path, num_clients: int, device: torch.device):
    cid_int = int(cid)
    cases = list_cases_raw(pre_root)
    my_cases = partition_cases(cases, num_clients=num_clients, cid=cid_int)

    if cid_int == 0 and len(my_cases) > 0:
        import nibabel as nib
        import numpy as np
        seg0 = nib.load(str(my_cases[0].seg)).get_fdata().astype(np.int16)
        u0 = np.unique(seg0)
        print(f"[Sanity] raw seg labels (case {my_cases[0].case_id}): {u0}")

        # After remap happens in dataset, you can also check via dataset itself:
        # (This confirms the remap+transpose path)
        tmp_ds = BratsPatchDataset([my_cases[0]], patches_per_case=1, crop=(64, 64, 64), seed=123)
        _, y_patch = tmp_ds[0]
        print(f"[Sanity] patch labels after remap: {torch.unique(y_patch).cpu().numpy()}")

    # Simple split within each client: 80/20 train/val by case
    split = max(1, int(0.8 * len(my_cases)))
    train_cases = my_cases[:split]
    val_cases = my_cases[split:] if len(my_cases) > 1 else my_cases

    train_ds = BratsPatchDataset(train_cases, patches_per_case=2, crop=(64, 64, 64), seed=42 + cid_int)
    val_ds   = BratsPatchDataset(val_cases, patches_per_case=1, crop=(64, 64, 64), seed=999 + cid_int)
    case_val_ds = BratsCaseDataset(val_cases)

    trainloader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    valloader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    case_valloader = DataLoader(case_val_ds, batch_size=1, shuffle=False, num_workers=0)

    return BratsClient(
        cid=cid_int,
        trainloader=trainloader,
        valloader=valloader,
        case_valloader=case_valloader,
        device=device,
        lr=1e-3,
        local_epochs=1,
    ).to_client()


def main():
    pre_root = Path("data/brats_raw")
    num_clients = 2
    num_rounds = 20
    local_epochs = 2

    device = torch.device("cpu")

    strategy = make_strategy(num_clients=num_clients, local_epochs=local_epochs)

    hist = fl.simulation.start_simulation(
        client_fn=lambda cid: client_fn(cid, pre_root=pre_root, num_clients=num_clients, device=device),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 16, "num_gpus": 0},
    )
    # ---- Save aggregated per-round results to CSV ----
    out_dir = Path("repro/flwr")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"fedavg_sim_{num_clients}clients_{num_rounds}rounds_{stamp}.csv"

    # hist.losses_distributed: List[Tuple[round, loss]]
    loss_by_round = {rnd: loss for rnd, loss in (hist.losses_distributed or [])}

    # hist.metrics_distributed: Dict[str, List[Tuple[round, value]]]
    metrics_by_name = hist.metrics_distributed or {}
    metric_maps = {
        name: {rnd: val for rnd, val in series} for name, series in metrics_by_name.items()
    }

    rounds = range(1, num_rounds + 1)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["round", "loss", "dice_WT", "dice_TC", "dice_ET"])
        w.writeheader()
        for r in rounds:
            row = {
                "round": r,
                "loss": loss_by_round.get(r, ""),
                "dice_WT": metric_maps.get("dice_WT", {}).get(r, ""),
                "dice_TC": metric_maps.get("dice_TC", {}).get(r, ""),
                "dice_ET": metric_maps.get("dice_ET", {}).get(r, ""),
            }
            w.writerow(row)

    print(f"Saved results CSV: {csv_path}")

if __name__ == "__main__":
    main()