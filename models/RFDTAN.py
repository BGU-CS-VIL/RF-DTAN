"""
Created on Oct  2019

author: ronsha
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from difw import Cpab
from tsai.all import *
from models.backbones.cnn import get_locnet

# from TSAI
MODEL_DICT = {
    "InceptionTime": InceptionTime,
    # TCN from TSAI
    "TCN": TCN,
    # CNN from DTAN (2019)
    "CNN": get_locnet,
}


class RFDTAN(nn.Module):
    """
    PyTroch nn.Module implementation of Regularization-free Diffeomorphic Temporal Alignment Nets
    """

    def __init__(
        self,
        signal_len,
        channels,
        tess=[16,],
        n_recurrence=1,
        zero_boundary=True,
        device="cuda",
        backbone="InceptionTime",
        **kwargs,
    ):
        """
        Initialize the DTAN model.
        
        Args:
        - signal_len (int): signal length.
        - channels (int): number of channels.
        - tess (list): tessellation size.
        - n_recurrence (int): Number of recurrences for R-DTAN.
                              Increasing the number of recurrences does not increase the number of parameters,
                              but does increase the training time. Default is 1.
        - zero_boundary (bool): Zero boundary (when True) for input X and transformed version X_T.
                                It sets X[0]=X_T[0] and X[n] = X_T[n]. Default is True.
        - device (str): Device to use. 'gpu' or 'cpu'.
        - backbone (str): Backbone architecture. Default is 'InceptionTime'.
        """

        super().__init__()

        # init CPAB transformer
        cpab_device = "cpu" if device == "cpu" else "gpu"
        self.T = Cpab(
            tess, backend="pytorch", device=cpab_device, zero_boundary=zero_boundary
        )
        self.theta_dim = self.T.params.d
        self.output_dim = 64
        self.alignment_head_dim = self.theta_dim
        self.n_recurrence = n_recurrence
        self.sz = signal_len  # signal length
        self.n_ss = 8  # squaring and scaling factor
        self.channels = channels
        self.device = device
        self.backbone = backbone
        self.dropout = nn.Dropout(0.1)

        assert backbone in MODEL_DICT.keys(), f"{backbone} not in {MODEL_DICT.keys()}"
        self.arch = MODEL_DICT[backbone]
        if self.backbone == "CNN":
            self.backbone = self.arch(channels, output_dim=self.output_dim)
        else:
            self.backbone = build_ts_model(
                arch=self.arch, c_in=channels, c_out=self.output_dim, seq_len=signal_len
            )

        # HEADs
        self.alignment_head = nn.Sequential(
            nn.Linear(self.output_dim, self.alignment_head_dim)
        )
        nn.init.normal_(self.alignment_head[-1].weight, std=1e-5)
        nn.init.normal_(self.alignment_head[-1].bias, std=1e-5)

    def ttn(self, x, embedding, return_theta=False):
        """
        Apply Temporal transformer network to the input x.
        
        Args:
        - x (torch.Tensor): Input signal with shape (nb, channels, ts).
        - embedding (torch.Tensor): Embedding.
        - return_theta (bool): Flag indicating whether to return theta values.
        
        Returns:
        - x (torch.Tensor): Transformed signal.
        - theta (torch.Tensor, optional): Theta values (if return_theta is True).
        """

        embedding = self.dropout(embedding)
        theta = self.alignment_head(embedding)
        theta = theta.view(-1, 1, self.theta_dim)
        channels, sz = x.shape[1:]
        # transform data needs channel last
        x = torch.reshape(x, (-1, sz, channels))
        # (N, C, sz)
        x = self.transform_data(x, theta)
        theta = theta.view(-1, self.theta_dim)

        if not return_theta:
            return x
        else:
            return x, theta

    def forward(self, x, return_nans=False):
        """
        Perform forward pass on the input signal x and return the transformed signal.
        
        Args:
        - x (torch.Tensor): Input signal with shape (nb, channels, ts).
        - return_nans (bool): Flag indicating whether to return the nan mask.
        
        Returns:
        - outputs (dict): Dictionary containing the following keys:
            - "data" (torch.Tensor): Transformed signal.
            - "nans_mask" (torch.Tensor): Nan mask.
            - "embedding" (torch.Tensor): Embedding.
            - "theta" (torch.Tensor, optional): Theta values.
        """

        outputs = {}
        nb, channels, sz = x.shape
        nr = self.n_recurrence
        thetas = torch.zeros((nb, nr, self.theta_dim), device=x.device)
        has_nans = torch.any(torch.isnan(x))

        # Variable Length
        if has_nans:
            nan_mask = torch.isnan(x).float().reshape(-1, sz, channels).to(device)
            x = torch.nan_to_num(x, nan=0)

        xt = torch.clone(x)

        for i in range(self.n_recurrence):
            embedding = self.backbone(xt)
            xt, theta = self.ttn(xt, embedding, return_theta=True)
            thetas[:, i, :] = theta

        x = self.transform_data(torch.reshape(x, (-1, sz, channels)), thetas)

        if has_nans:
            with torch.no_grad():
                nan_mask = self.transform_data(nan_mask, thetas)
                nan_mask = nan_mask.bool()
        else:
            nan_mask = torch.zeros((nb, channels, sz), device=x.device).bool()

        if return_nans:
            x[nan_mask] = torch.nan

        outputs["data"] = x
        outputs["nans_mask"] = nan_mask
        outputs["embedding"] = embedding
        outputs["theta"] = thetas

        return outputs


    def get_basis(self):
        """
        Get the basis used for transformation.
        
        Returns:
        - self.T (Cpab): Basis used for transformation.
        """
        return self.T

    def set_recurrences(self, n_recurrence: int):
        """
        Set the number of recurrences for R-DTAN.
        
        Args:
        - n_recurrence (int): Number of recurrences.
        """
        self.n_recurrence = n_recurrence

    def get_conv_to_fc_dim(self):
        """ 
        Calculate the output dimension of the backbone when using the simple CNN.
        
        Returns:
        - conv_to_fc_dim (int): Dimension after convolutional layers.
        """
        rand_tensor = torch.rand([1, self.channels, self.sz])
        out_tensor = self.backbone(rand_tensor)
        conv_to_fc_dim = out_tensor.size(1) * out_tensor.size(2)
        return conv_to_fc_dim
 
    def transform_grid(self, theta):
        """
        Transform the grid using the given theta values.
        
        Args:
        - theta (torch.Tensor): Theta values with shape (nb, nr, theta_dim).
        
        Returns:
        - output_grid (torch.Tensor): Transformed grid with shape (nb, sz).
        """

        nb, nr, theta_dim = theta.shape
        output_grid = torch.zeros((nb, self.sz), device=self.device)

        for i in range(nr):
            grid = self.get_grid(n_grids=theta.shape[0])
            theta_i = theta[:, i, :] / 2 ** (self.n_ss)
            grid_t = self.T.transform_grid_ss(
                grid, theta_i, method="closed_form", N=self.n_ss, time=1
            )
            delta_grid = grid_t - grid
            output_grid += delta_grid

        output_grid += grid
        return output_grid

    def get_grid(self, n_grids=1):
        """
        Get the grid for transformation.
        
        Args:
        - n_grids (int): Number of grids to generate.
        
        Returns:
        - grid (torch.Tensor): Grid with shape (n_grids, sz).
        """
        grid = self.T.uniform_meshgrid(n_points=self.sz).repeat(n_grids, 1)
        return grid

    def transform_data(self, x, theta, reshape=True):
        """
        Transform the input data x using the given theta values.
        
        Args:
        - x (torch.Tensor): Input data with shape (nb, ts, channels).
        - theta (torch.Tensor): Theta values with shape (nb, nr, theta_dim).
        - reshape (bool): Flag indicating whether to reshape the output.
        
        Returns:
        - xt (torch.Tensor): Transformed data with shape (nb, channels, sz).
        """

        grid_t = self.transform_grid(theta)
        xt = self.T.interpolate(x, grid_t, outsize=self.sz)

        if reshape:
            xt = torch.reshape(xt, (-1, self.channels, self.sz))

        return xt

    def transform_data_inverse(self, x, theta):
        """ 
        Inverse transform the input data x using the given theta values.
        
        Args:
        - x (torch.Tensor): Input data with shape (nb, ts, channels).
        - theta (torch.Tensor): Theta values with shape (nb, nr, theta_dim).
        
        Returns:
        - xt (torch.Tensor): Inverse transformed data with shape (nb, channels, sz).
        """

        if self.n_recurrence > 1:
            theta = torch.flip(theta, dims=[1])
        xt = self.transform_data(x, -1 * theta)
        return xt
