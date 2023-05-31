

import torch
import os
from loss.alignment_loss import compute_loss_dict
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def update_loss_dict(epoch_loss_dict: dict, batch_loss_dict: dict, N: int) -> dict:
    """
    Update the epoch loss dictionary with batch loss values.

    Args:
        epoch_loss_dict (dict): Dictionary to store the epoch loss values.
        batch_loss_dict (dict): Dictionary containing the batch loss values.
        N (int): Batch size.

    Returns:
        dict: Updated epoch loss dictionary.

    """
    for key, loss_val in batch_loss_dict.items():
        if isinstance(loss_val, torch.Tensor):
            loss_val = loss_val.detach().cpu().item()
        epoch_loss_dict[key] += loss_val / N

    return epoch_loss_dict


def train_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    device: str,
    optimizer: Optimizer,
    loss_func_dict: dict,
    weights=None,
) -> dict:
    """
    Perform one training epoch.

    Args:
        dataloader (DataLoader): Torch dataloader.
        model (nn.Module): Torch model.
        device (str): Device to be used for computation.
        optimizer (Optimizer): Torch optimizer.
        loss_func_dict (dict): Dictionary of loss functions.
        weights (dict, optional): Dictionary of weights for each loss function.

    Returns:
        dict: Dictionary containing the epoch loss values.

    """
    model.train()
    epoch_loss_dict = {k: 0 for k in loss_func_dict.keys()}
    epoch_loss_dict["loss"] = 0
    for X, y in dataloader:
        optimizer.zero_grad()
        N = X.shape[0]
        batch_loss_dict = one_batch(X, y, model, device, loss_func_dict, weights)
        loss = batch_loss_dict["loss"]
        loss.backward()
        optimizer.step()
        epoch_loss_dict = update_loss_dict(epoch_loss_dict, batch_loss_dict, N)
        
    return epoch_loss_dict


def val_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    device: torch.device,
    loss_func_dict: dict,
    weights=None,
) -> dict:
    """
    Perform one validation epoch.

    Args:
        dataloader (DataLoader): Torch dataloader.
        model (nn.Module): Torch model.
        device (torch.device): Device to be used for computation.
        loss_func_dict (dict): Dictionary of loss functions.
        weights (dict, optional): Dictionary of weights for each loss function.

    Returns:
        dict: Dictionary containing the epoch loss values.

    """
    epoch_loss_dict = {k: 0 for k in loss_func_dict.keys()}
    epoch_loss_dict["loss"] = 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            N = X.shape[0]
            batch_loss_dict = one_batch(X, y, model, device, loss_func_dict, weights)
            epoch_loss_dict = update_loss_dict(epoch_loss_dict, batch_loss_dict, N)
            loss = batch_loss_dict["loss"]

    return epoch_loss_dict


def one_batch(
    X: torch.Tensor,
    y: torch.Tensor,
    model: nn.Module,
    device: str,
    loss_func_dict: dict,
    weights=None,
) -> dict:
    """
    Perform computations for one batch.

    Args:
        X: Input tensor.
        y: Label tensor.
        model (nn.Module): Torch model.
        device (torch.device): Device to be used for computation.
        loss_func_dict (dict): Dictionary of loss functions.
        weights (dict, optional): Dictionary of weights for each loss function.

    Returns:
        dict: Dictionary containing the batch loss values.

    """
    X, y = X.to(device), y.to(device)
    outputs_dict = model(X)
    Xt, thetas = outputs_dict["data"], outputs_dict["theta"]

    loss_args_dict = {
        "X": X, 
        "Xt": Xt, 
        "y_true": y,
        "thetas": thetas,
        "nans_mask": outputs_dict["nans_mask"],
        "model": model,
    }

    batch_loss_dict = compute_loss_dict(loss_args_dict, loss_func_dict, weights=weights)

    return batch_loss_dict


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    val_loss: float,
    exp_name='',
    ckpt_dir='./checkpoints'
):
    """
    Save model checkpoint.

    Args:
        model (nn.Module): Torch model.
        optimizer (Optimizer): Torch optimizer.
        val_loss (float): Validation loss at the time of saving the checkpoint.
        exp_name (str): File name.
        ckpt_dir (str): Checkpoint directory.

    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': val_loss
    }

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    fullpath = os.path.join(ckpt_dir, exp_name)
    torch.save(checkpoint, f'{fullpath}_checkpoint.pth')
