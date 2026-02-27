from __future__ import annotations

from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import flwr as fl

from .model import UNet3D
from .metrics import brats_dice_regions


def get_parameters(net: torch.nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    state_dict = net.state_dict()
    keys = list(state_dict.keys())
    new_state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    net.load_state_dict(new_state, strict=True)


@torch.no_grad()
def evaluate_full_volume(net: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    For simplicity, we evaluate on the same patch loader by aggregating predictions over patches.
    This is a baseline; later you can implement sliding-window full-volume inference.
    """
    net.eval()
    all_pred = []
    all_gt = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = net(xb)  # (B, C, D, H, W)
        pred = torch.argmax(logits, dim=1).cpu().numpy()  # (B, D, H, W)
        gt = yb.numpy()
        all_pred.append(pred.reshape(-1))
        all_gt.append(gt.reshape(-1))

    pred_flat = np.concatenate(all_pred, axis=0)
    gt_flat = np.concatenate(all_gt, axis=0)
    return brats_dice_regions(pred_flat, gt_flat)


class BratsClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: int,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        lr: float = 1e-3,
        local_epochs: int = 1,
    ):
        self.cid = cid
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.local_epochs = local_epochs

        self.net = UNet3D(in_channels=4, num_classes=5, base=16).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()

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
        ce = 0.0
        n_batches = 0
        with torch.no_grad():
            for xb, yb in self.valloader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = self.net(xb)
                loss = self.criterion(logits, yb)
                ce += float(loss.item())
                n_batches += 1
        loss_avg = ce / max(1, n_batches)

        # Dice metrics (patch-based baseline)
        metrics = evaluate_full_volume(self.net, self.valloader, self.device)

        num_examples = len(self.valloader.dataset)
        return loss_avg, num_examples, metrics