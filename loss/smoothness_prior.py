"""
Created on Oct  2019
author: ronsha
"""

# From other libraries
import numpy as np
import torch.nn
from torch import Tensor

import torch


"""
Smoothness Norm:
Computes the smoothness and L2 norm of theta for cpab transformer T. 
1. Smoothness prior: penalize low correlation of theta between close cells in the tessellation. 
   It is computed by building a (D,D) covariance correlations decay with inter-cell distances.

2. L2 norm: penalize large values of theta
Arguments:
    theta: current parametrization of the transformation.
    T: cpab class of transformer type
    scale_spatial: smoothness regularization strength
    scale_value: L2 regularization strength
    print_info: for debugging, will probably be removed. 
Returns:
    Smoothness norm: high values indicates lack of smoothness. To be added to the loss as a regularization. 
"""



class PriorLoss(torch.nn.Module):
    
    def __init__(self, T, lambda_smooth=0.5, lambda_var=0.1, print_info=False) -> None:
            #super().__init__()

            super(PriorLoss, self).__init__()
            self.T = T
            self.lambda_smooth = lambda_smooth
            self.lambda_var = lambda_var
            self.precision_cov = self.get_precision_cov()
            
            self.print_info = print_info



    def torch_dist_mat(self, centers):
        '''
        Produces an NxN  dist matrix D,  from vector (centers) of size N
        Diagnoal = 0, each entry j, represent the distance from the diagonal
        dictated by the centers vector input
        '''
        times = centers.shape  # Torch.Tensor([n], shape=(1,), dtype=int32)

        # centers_grid tile of shape N,N, each row = centers
        centers_grid = centers.repeat(times[0],1)
        dist_matrix = torch.abs(centers_grid - torch.transpose(centers_grid, 0, 1))
        return dist_matrix


    #def forward(self, theta: Tensor) -> Tensor:
    def get_precision_cov(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        D, d = self.T.params.D, self.T.params.d
        B = self.T.params.B
        if isinstance(B, np.ndarray):
            B = torch.from_numpy(B).float()
            B = B.to(device)

        # Convert type
        #B = tf.cast(B, tf.float32)


        # Distance between centers
        centers = torch.linspace(-1., 1., D).to(device)  # from 0 to 1 with nC steps

        # calculate the distance
        dists = self.torch_dist_mat(centers)  # DxD

        # # scale the distance
        # for x>0, e^(-x^2) decays very fast from 1 to 0

        cov_avees = torch.exp(-(dists / self.lambda_smooth))
        cov_avees *= (cov_avees * (self.lambda_var * D) ** 2)

        B_T = torch.transpose(B, 0, 1)
        cov_cpa = torch.matmul(B_T, torch.matmul(cov_avees, B))
        eps = (torch.eye(cov_cpa.shape[0])*torch.finfo(torch.float32).eps).to(device)
        precision_theta = torch.inverse(cov_cpa + eps)
        return precision_theta



    # Domain space is [0,1]^dim where 0.5 is the origin
    def compute_norm(self, theta):
        theta_T = torch.transpose(theta, 0, 1)
        smooth_norm = torch.matmul(theta, torch.matmul(self.precision_cov, theta_T))
        smooth_norm = torch.mean(smooth_norm)

        return smooth_norm

    def forward(self, thetas=None, n_channels=1, **kwargs):
        """
            thetas: list of dictionaries of theta
        """
        loss = 0.
        for theta in thetas:

            for c in range(n_channels): 
                # alignment loss takes over variance loss
                # larger penalty when k increases -> coarse to fine
                if len(theta.shape) == 2:
                    theta = torch.unsqueeze(theta, dim=1)

                loss += 0.1*(1/n_channels)*self.compute_norm(theta[:, c, :])

        return loss

    def set_labmda_var(self, value):
        self.lambda_var = value
    
    def set_lambda_smooth(self, value):
        self.lambda_smooth = value

    def get_labmdas(self):
        return self.lambda_smooth, self.lambda_var