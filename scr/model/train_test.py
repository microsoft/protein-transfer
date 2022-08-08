"""A script with model training and testing details"""

from __future__ import annotations

import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader 
from tqdm import tqdm

from scr.params.sys import RAND_SEED, DEVICE

# seed everything
random.seed(RAND_SEED)
np.random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)
torch.cuda.manual_seed(RAND_SEED)
torch.cuda.manual_seed_all(RAND_SEED)
torch.backends.cudnn.deterministic = True

def run_epoch(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device | str = DEVICE,
        criterion: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None
        ) -> float:
    
    """
    Runs one epoch.
    
    Args:
    - model: nn.Module, already moved to device
    - loader: torch.utils.data.DataLoader
    - device: torch.device or str
    - criterion: optional nn.Module, loss function, already moved to device
    - optimizer: optional torch.optim.Optimizer, must also provide criterion,
        only provided for training

    Returns: 
    - float, average loss over batches
    """
    if optimizer is not None:
        assert criterion is not None
        model.train()
        is_train = True
    else:
        model.eval()
        is_train = False

    cum_loss = 0.

    with torch.set_grad_enabled(is_train):
        for (x, y, _, _, _) in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            outputs = model(x)

            if criterion is not None:
                loss = criterion(outputs, y.float())

                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                cum_loss += loss.item()

    return cum_loss / len(loader)

def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          criterion: nn.Module,
          device: torch.device | str = DEVICE,
          learning_rate: float = 1e-4,
          lr_decay: float = 0.1,
          epochs: int = 100,
          early_stop: bool = True,
          tolerance: int = 10,
          min_epoch: int = 5,
          ) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
    - model: nn.Module, already moved to device
    - train_loader: torch.utils.data.DataLoader, 
    - val_loader: torch.utils.data.DataLoader, 
    - criterion: nn.Module, loss function, already moved to device
    - device: torch.device or str
    - learning_rate: float
    - lr_decay: float, factor by which to decay LR on plateau
    - epochs: int, number of epochs to train for
    - early_stop: bool = True,

    Returns: 
    - tuple of np.ndarray, (train_losses, val_losses)
        train/val_losses: np.ndarray, shape [epochs], entries are average loss
        over batches for that epoch
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_decay)

    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)

    # init for early stopping
    counter = 0
    min_val_loss = np.Inf

    for epoch in tqdm(range(epochs)):
        train_losses[epoch] = run_epoch(
            model=model, loader=train_loader, device=device,
            criterion=criterion, optimizer=optimizer)

        val_loss = run_epoch(
            model=model, loader=val_loader, device=device,
            criterion=criterion)
        val_losses[epoch] = val_loss

        scheduler.step(val_loss)

        if early_stop:
            # when val loss decrease, reset min loss and counter
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                counter = 0
            else:
                counter += 1

            if epoch > min_epoch and counter == tolerance:
                break

    return train_losses, val_losses


def test(model: nn.Module,
         loader: DataLoader,
         criterion: nn.Module | None,
         device: torch.device | str = DEVICE,
         print_every: int = 1000,
         ) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Runs one epoch of testing, returning predictions and labels.
    
    Args:
    - model: nn.Module, already moved to device
    - device: torch.device or str
    - loader: torch.utils.data.DataLoader
    - criterion: optional nn.Module, loss function, already moved to device
    - print_every: int, how often (number of batches) to print avg loss
    
    Returns: tuple (avg_loss, preds, labels)
    - avg_loss: float, average loss per training example 
    - preds: np.ndarray, shape [num_examples, ...], predictions over dataset
    - labels: np.ndarray, shape [num_examples, ...], dataset labels
    """
    model.eval()
    msg = "[{step:5d}] loss: {loss:.3f}"

    cum_loss = 0.0

    preds = []
    labels = []

    with torch.no_grad():

        for i, (x, y, _, _, _) in enumerate(tqdm(loader)):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # forward + backward + optimize
            outputs = model(x)
            preds.append(outputs.detach().cpu().squeeze().numpy())
            labels.append(y.detach().cpu().squeeze().numpy())

            if criterion is not None:
                loss = criterion(outputs, y)
                cum_loss += loss.item()

                if ((i + 1) % print_every == 0) or (i + 1 == len(loader)):
                    tqdm.write(msg.format(step=i + 1, loss=cum_loss / len(loader)))

    avg_loss = cum_loss / len(loader)
    return avg_loss, np.concatenate(preds), np.concatenate(labels)
