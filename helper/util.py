"""
Created on Oct 2019

Author: ronsha
"""

import numpy as np
import torch
import torch.nn.functional as F


def align_dataset(dataloader, model, device=None, return_theta=False):
    """
    Align the dataset using the provided model.
    
    Args:
    - dataloader: Torch dataloader.
    - model: Model used for alignment.
    - device: Device to use.
    - return_theta: Flag indicating whether to return theta values.
    
    Returns:
    - aligned_data: Aligned data.
    - thetas: Theta values.
    - targets: Target labels.
    - nan_masks: Nan masks.
    """

    model.eval()

    aligned_data = []
    targets = []  # in case dataloader is shuffled
    thetas = []
    nan_masks = []

    for batch_idx, (data, target) in enumerate(dataloader):

        if device is not None:
            data, target = data.to(device), target.to(device)

        outputs_dict = model(data)
        output, thetas_out = outputs_dict['data'], outputs_dict['theta']
        nan_mask = outputs_dict['nans_mask']

        aligned_data.append(output.detach().cpu())
        targets.append(target.detach().cpu())
        nan_masks.append(nan_mask.detach().cpu())
        thetas_list = [t.detach().cpu() for t in thetas_out]

        thetas += thetas_list  # append theta to the list

        del data, target, outputs_dict, output

    if return_theta:
        thetas = torch.stack(thetas)

    aligned_data = torch.vstack(aligned_data)
    targets = torch.hstack(targets)
    nan_masks = torch.vstack(nan_masks)

    return aligned_data, thetas, targets, nan_masks


def get_dataset_info(dataset_name, X_train, X_test, y_train, y_test, print_info=True):
    """
    Get information about the dataset.
    
    Args:
    - dataset_name: Name of the dataset.
    - X_train: Training data.
    - X_test: Testing data.
    - y_train: Training labels.
    - y_test: Testing labels.
    - print_info: Flag indicating whether to print dataset information.
    
    Returns:
    - input_shape: Shape of the input data.
    - n_classes: Number of classes in the dataset.
    """

    N, C, input_shape = X_train.shape
    n_classes = len(np.unique(y_train))

    if print_info:
        print(f"{dataset_name} dataset details:")
        print('    X train.shape:', X_train.shape)
        print('    X test.shape:', X_test.shape)
        print('    y train.shape:', y_train.shape)
        print('    y test.shape:', y_test.shape)
        print('    number of classes:', n_classes)
        print('    number of (train) samples:', N)
        print('    number of channels:', C)
        print('    input shape:', input_shape)

    return input_shape, n_classes
