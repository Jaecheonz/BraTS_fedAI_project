# client.py
from __future__ import annotations

from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import flwr as fl

from .model import UNet3D
from .metrics import brats_dice_regions

from monai.inferers import sliding_window_inference


def get_parameters(net: torch.nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    state_dict = net.state_dict()
    keys = list(state_dict.keys())
    new_state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    net.load_state_dict(new_state, strict=True)


@torch.no_grad()
def evaluate_cases_sliding_window(
    net: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    roi_size: Tuple[int, int, int] = (96, 96, 96),
    sw_batch_size: int = 1,
) -> Dict[str, float]:
    """
    True case-level evaluation using sliding-window inference over full volumes.
    Expects loader batches of: x (B,C,D,H,W), y (B,D,H,W), case_id (B,)
    """
    net.eval()
    dice_wt, dice_tc, dice_et = [], [], []

    for xb, yb, _case_id in loader:
        xb = xb.to(device)  # (B,C,D,H,W)
        y = yb.cpu().numpy()  # (B,D,H,W)

        # Sliding window returns logits: (B,C,D,H,W)
        logits = sliding_window_inference(xb, roi_size=roi_size, sw_batch_size=sw_batch_size, predictor=net)
        pred = torch.argmax(logits, dim=1).cpu().numpy()  # (B,D,H,W)

        # Compute dice per case (B should be 1, but handle general B)
        for i in range(pred.shape[0]):
            # --- DEBUG: ET presence (class 3 after remap) ---
            gt_et = int((y[i] == 3).sum())
            pr_et = int((pred[i] == 3).sum())
            gt_wt = int((y[i] > 0).sum())
            pr_wt = int((pred[i] > 0).sum())
            case_id = _case_id[i] if isinstance(_case_id, (list, tuple, np.ndarray)) else _case_id
            print(f"[ET dbg] case={case_id} gt_ET={gt_et} pred_ET={pr_et} gt_WTvox={gt_wt} pred_WTvox={pr_wt}")

            m = brats_dice_regions(pred[i].reshape(-1), y[i].reshape(-1))
            dice_wt.append(m["dice_WT"])
            dice_tc.append(m["dice_TC"])
            dice_et.append(m["dice_ET"])

    # Average, ignoring NaNs (your dice_bool uses NaN for empty-empty)
    def nanmean(xs):
        xs = np.array(xs, dtype=np.float32)
        return float(np.nanmean(xs)) if np.any(~np.isnan(xs)) else float("nan")

    return {
        "dice_WT": nanmean(dice_wt),
        "dice_TC": nanmean(dice_tc),
        "dice_ET": nanmean(dice_et),
    }


class BratsClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: int,
        trainloader: DataLoader,
        valloader: DataLoader,
        case_valloader: DataLoader,
        device: torch.device,
        lr: float = 1e-3,
        local_epochs: int = 1,
    ):
        self.cid = cid
        self.trainloader = trainloader
        self.valloader = valloader
        self.case_valloader = case_valloader
        self.device = device
        self.local_epochs = local_epochs

        self.net = UNet3D(in_channels=4, num_classes=4, base=16).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        w = torch.tensor([1.0, 2.0, 2.0, 6.0], device=self.device)  # [bg, class1, class2, ET]
        self.criterion = torch.nn.CrossEntropyLoss(weight=w)

    def get_parameters(self, config: Dict[str, Any]):
        return get_parameters(self.net)

    def fit(self, parameters, config: Dict[str, Any]):
        set_parameters(self.net, parameters)
        self.net.train()

        local_epochs = int(config.get("local_epochs", self.local_epochs))

        for _ in range(local_epochs):
            for xb, yb in self.trainloader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)  # (B, D, H, W)

                self.optim.zero_grad()
                logits = self.net(xb)  # (B, C, D, H, W)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optim.step()

        num_examples = len(self.trainloader.dataset)
        return get_parameters(self.net), num_examples, {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)

        # Compute CE loss on val patches
        self.net.eval()
        loss_sum = 0.0
        n_batches = 0
        with torch.no_grad():
            for xb, yb in self.valloader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = self.net(xb)
                loss = self.criterion(logits, yb)
                loss_sum += float(loss.item())
                n_batches += 1
        loss_avg = loss_sum / max(1, n_batches)

        # Dice metrics (patch-based baseline)
        rnd = int(config.get("rnd", 0))
        if rnd % 2 == 1:
            metrics = evaluate_cases_sliding_window(self.net, self.case_valloader, self.device, roi_size=(64, 64, 64), sw_batch_size=1)
            num_examples = len(self.case_valloader.dataset)   # <-- use cases
        else:
            metrics = {"dice_WT": float("nan"), "dice_TC": float("nan"), "dice_ET": float("nan")}
            num_examples = len(self.valloader.dataset)        # patches (loss still patch-based)

        return loss_avg, num_examples, metrics