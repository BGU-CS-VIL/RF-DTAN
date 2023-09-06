# local
from helper.util import get_dataset_info

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
from sklearn.model_selection import train_test_split
#from tsai.data.external import get_UCR_data as get_UCR_data_tsai
from tslearn.datasets import UCR_UEA_datasets

import numpy as np


class UCRDataset(Dataset):
    """
    Dataset class for UCR time series data.
    X - (N, C, sz) numpy array of time series data.
    y - (N,) numpy array of labels.
    """

    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = X
        self.y = y

    def __getitem__(self, index):        
        return self.X[index], self.y[index]
    def __len__(self):
        return len(self.X)

    def set_y(self, y):
        self.y = y

    def set_X(self, X):
        self.X = X



def np_to_dataset(X, y, onehot=False):
    """
    Convert numpy arrays to PyTorch dataset.

    Args:
    - X: Input data as numpy array.
    - y: Target labels as numpy array.
    - onehot: Flag indicating whether the labels should be one-hot encoded.

    Returns:
    - dataset: PyTorch dataset.
    """

    X_tensor = torch.Tensor(X)
    y_tensor = torch.Tensor(y)
    y_tensor = y_tensor.long()

    if onehot:
        y_tensor = y_tensor.float()

    dataset = UCRDataset(X_tensor, y_tensor)

    return dataset


def get_train_and_validation_loaders(
    dataset, validation_split=0.1, batch_size=32, shuffle=True, rand_seed=42
):
    """
    Get train and validation loaders for the dataset.

    Args:
    - dataset: PyTorch dataset.
    - validation_split: Fraction of the data to be used for validation. Default is 0.1.
    - batch_size: Batch size. Default is 32.
    - shuffle: Flag indicating whether to shuffle the data. Default is True.
    - rand_seed: Random seed for shuffling. Default is 42.

    Returns:
    - train_loader: DataLoader for training data.
    - validation_loader: DataLoader for validation data.
    """

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    X, y = dataset[:]

    train_indices, val_indices = train_test_split(
        indices, test_size=validation_split, random_state=rand_seed, stratify=y
    )

    if shuffle:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
    else:
        train_sampler = SequentialSampler(train_indices)
        valid_sampler = SequentialSampler(val_indices)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True
    )
    validation_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler, pin_memory=True
    )

    return train_loader, validation_loader


def processed_UCR_data(
    X_train,
    y_train,
    X_test,
    y_test,
    onehot=False,
    swap_channel_dim=False,
    replace_nans=True,
    resample=-1,
):
    """
    Process UCR data by normalizing, fixing labels, and adding channel dimensions if necessary.

    Args:
    # Assumes tslearn data format - (N, sz, C). Torch needs (N, C, sz)
    - X_train: Training data as numpy array.
    - y_train: Training labels as numpy array.
    - X_test: Testing data as numpy array.
    - y_test: Testing labels as numpy array.
    - onehot: Flag indicating whether the labels should be one-hot encoded. Default is False.
    - swap_channel_dim: Flag indicating whether to swap the channel dimension. Default is False.
    - replace_nans: Flag indicating whether to replace NaN values with zeros. Default is True.
    - resample: Size to resample the time series data to. Default is -1 (no resampling).

    Returns:
    - X_train: Processed training data as numpy array.
    - y_train: Processed training labels as numpy array.
    - X_test: Processed testing data as numpy array.
    - y_test: Processed testing labels as numpy array.
    """

    scaler = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0)

    for c in range(X_train.shape[-1]):
        X_train[:, :, c] = np.squeeze(scaler.fit_transform(X_train[:, :, c]))
        X_test[:, :, c] = np.squeeze(scaler.fit_transform(X_test[:, :, c]))

    if resample > 0:
        X_train = TimeSeriesResampler(sz=resample).fit_transform(np.squeeze(X_train))
        X_test = TimeSeriesResampler(sz=resample).fit_transform(np.squeeze(X_test))

    if len(X_train.shape) < 3:
        X_train = np.expand_dims(X_train, 1)
        X_test = np.expand_dims(X_test, 1)

    class_names = np.unique(y_train, axis=0)
    y_train_tmp = np.zeros(len(y_train))
    y_test_tmp = np.zeros(len(y_test))

    for i, class_name in enumerate(class_names):
        y_train_tmp[y_train == class_name] = i
        y_test_tmp[y_test == class_name] = i

    y_train = y_train_tmp
    y_test = y_test_tmp

    if swap_channel_dim:
        X_train = np.swapaxes(X_train, 2, 1)
        X_test = np.swapaxes(X_test, 2, 1)

    if onehot:
        y_train = np.eye(len(class_names))[y_train.astype(int)].astype(float)
        y_test = np.eye(len(class_names))[y_test.astype(int)].astype(float)

    if replace_nans:
        if np.isnan(X_train).any():
            X_train = np.nan_to_num(X_train)
        if np.isnan(X_test).any():
            X_test = np.nan_to_num(X_test)

    return X_train, y_train, X_test, y_test


def get_UCR_data(
    dataset_name,
    data_dir,
    batch_size=32,
    val_split=0.2,
    onehot=False,
    swap_channel_dim=False,
    replace_nans=False,
):
    """
    Load UCR time series data.

    Args:
    - dataset_name: Name of the dataset.
    - data_dir: Directory containing the UCR data.
    - batch_size: Batch size. Default is 32.
    - val_split: Fraction of the data to be used for validation. Default is 0.2.
    - onehot: Flag indicating whether the labels should be one-hot encoded. Default is False.
    - swap_channel_dim: Flag indicating whether to swap the channel dimension. Default is False.
    - replace_nans: Flag indicating whether to replace NaN values with zeros. Default is False.

    Returns:
    - train_dataloader: DataLoader for training data.
    - validation_dataloader: DataLoader for validation data.
    - test_dataloader: DataLoader for testing data.
    """
    # Numpy
    # X_train, y_train, X_test, y_test = get_UCR_data_tsai(
    #     dataset_name, parent_dir=data_dir
    # )

    UCR_data = UCR_UEA_datasets()
    # tslearn format (N, sz, C). Torch needs (N, C, sz)
    X_train, y_train, X_test, y_test = UCR_data.load_dataset(dataset_name)
    # Numpy
    X_train, y_train, X_test, y_test = processed_UCR_data(
        X_train,
        y_train,
        X_test,
        y_test,
        onehot,
        swap_channel_dim,
        replace_nans,
    )

    _ = get_dataset_info(
        dataset_name, X_train, X_test, y_train, y_test, print_info=True
    )
    # Torch
    train_dataset = np_to_dataset(X_train, y_train, onehot)
    if val_split > 0:
        train_dataloader, validation_dataloader = get_train_and_validation_loaders(
            train_dataset,
            validation_split=val_split,
            batch_size=batch_size,
            shuffle=True,
        )
    else:
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        validation_dataloader = None

    test_dataset = np_to_dataset(X_test, y_test, onehot)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, validation_dataloader, test_dataloader
