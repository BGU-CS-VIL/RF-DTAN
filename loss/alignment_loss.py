import torch
import torch.nn as nn
from torch.nn import MSELoss
import numpy as np 

mse_loss = MSELoss()


def compute_loss_dict(loss_args_dict: dict, loss_func_dict: dict, weights: dict = None):
    """
    Computes the loss dictionary given a dictionary of loss arguments and loss functions.

    Args:
        loss_args_dict (dict): A dictionary of loss arguments.
        loss_func_dict (dict): A dictionary of loss functions.
        weights (dict, optional): A dictionary of weights for each loss function. If None, equal weights are assigned.

    Returns:
        dict: A dictionary containing the computed losses.
    """
    loss_dict = {}
    loss_dict["loss"] = 0
    n_losses = len(loss_func_dict.keys())
    eval_metrics = ["accuracy"]

    if weights is None:
        weights = {}
        for key in loss_func_dict.keys():
            weights[key] = np.ones(n_losses) / n_losses

    # Compute the losses
    for loss_name, loss_func in loss_func_dict.items():
        if loss_name not in eval_metrics:
            loss_val = weights[loss_name] * loss_func(**loss_args_dict)
            loss_dict[loss_name] = loss_val
            loss_dict["loss"] += loss_val
        else:
            loss_val = loss_func(**loss_args_dict)
            loss_dict[loss_name] = loss_val

    return loss_dict



class WCSS(nn.Module):
    """
    Within Class Sum of Squares (WCSS) loss module.
    """

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(
        self,
        Xt: torch.Tensor = None,
        y_true: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute the WCSS loss.

        Args:
            Xt (torch.Tensor): Input tensor.
            y_true (torch.Tensor): True labels tensor.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: Computed loss.

        """

        classes = torch.unique(y_true)
        loss = 0

        for k in classes:
            Xk = Xt[y_true == k]
            X_mean = Xk.mean(axis=0)
            nk = Xk.shape[0]
            X_mean_stack = X_mean.repeat(nk, 1, 1)  # Repeat along the batch axis
            # Mean squared error (MSE) from the mean represents variance
            loss += mse_loss(Xk, X_mean_stack)

        loss = loss / len(classes)

        return loss


class WCSSTriplet(nn.Module):
    """
    WCSS Triplet Loss module for Torch.

    Args:
        margin (float): Margin value for the triplet loss.
        has_nans (bool): Indicates whether the input data contains NaN values.

    """

    def __init__(self, margin=1.0, has_nans=False):
        super().__init__()

        self.wcss_loss = WCSS()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2, reduction="mean")
        self.has_nans = has_nans

    def forward(
        self,
        X=None,
        Xt=None,
        y_true=None,
        nans_mask=None,
        **kwargs
    ):
        """
        Compute the WCSS triplet loss.

        Args:
            X (torch.Tensor): Input tensor.
            Xt (torch.Tensor): Transformed input tensor.
            y_true (torch.Tensor): True labels tensor.
            thetas (list[torch.Tensor]): List of theta tensors.
            model: The CPAB transformer model.
            nans_mask: Mask for NaN values.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: Computed loss.

        """

        loss = 0.0

        n_classes = torch.unique(y_true)
        nk = len(n_classes)

        for i in n_classes:
            Xt_k = Xt[y_true == i]
            X_k = X[y_true == i]
            mask_k = nans_mask[y_true == i]

            if nk > 1:
                neg_classes = n_classes[n_classes != i]
                i_neg = torch.tensor(np.random.choice(neg_classes, 1), device=X.device)
            else:
                print("Warning: only one class in batch")
                i_neg = i

            mask_sum = torch.logical_not(mask_k).sum(dim=0)
            mask_sum[mask_sum == 0.0] = 1.0
            X_mean = Xt_k.sum(dim=0) / mask_sum

            if nk > 1:
                mask_neg_k = nans_mask[y_true == i_neg]
                mask_neg_k = torch.logical_not(mask_neg_k).sum(dim=0)
                mask_neg_k[mask_neg_k == 0.0] = 1.0
                Xt_k_neg = Xt[y_true == i_neg]
                X_neg_mean = Xt_k_neg.sum(dim=0) / mask_neg_k

            n_samples = Xt_k.shape[0]
            X_mean_stack = X_mean.repeat(n_samples, 1, 1)

            loss += self.triplet_loss(Xt_k, X_mean_stack, X_neg_mean)


            loss += self.wcss_loss(Xt, y_true)

        return loss


class ICAE(nn.Module):
    """
    Torch data format is  [N, C, W] W=timesteps
    Args:
        X_trasformed:
        labels:loss
        thetas: list, len(thetas) = n_recurrences. thetas[i]: (nbatch, n_channels, theta_dim)
        T - CPAB transformer

    Returns:

    """

    def __init__(self, has_nans=False):
        super().__init__()
        # reduction='none'
        if not has_nans:
            self.loss = MSELoss(reduction='mean')
        else:
            self.loss = MSENANLoss()

    def forward(self, X=None, Xt=None, y_true=None, thetas: list[torch.tensor]=None,
                model=None, nans_mask=None, **kwargs):
        loss = 0.0

        n_classes = torch.unique(y_true)
        batchsize, channels, input_shape = Xt.shape
        for i in n_classes:
            Xt_k = Xt[y_true == i]
            X_k = X[y_true == i]
            mask_k = nans_mask[y_true==i]

            # (N, C, sz) -> (C,sz)
            mask_sum = torch.logical_not(mask_k).sum(dim=0)
            # if there are zeros
            mask_sum[mask_sum==0.] = 1.
            X_mean = Xt_k.sum(dim=0) / mask_sum

            n_samples = Xt_k.shape[0]
            X_mean_stack = X_mean.repeat(n_samples, 1, 1)
            theta_k = thetas[y_true == i]

            
            # transform inverse
            X_mean_stack_inv = model.transform_data_inverse(X_mean_stack, theta_k)
            # compute mse between mean signal and original ones
            # original ones might have nans
            loss += self.loss(X_mean_stack_inv, X_k)


            
        return loss


class ICAETriplet(nn.Module):
    """
    ICAE + Triplet Loss module for Torch.

    Args:
        margin (float): Margin value for the triplet loss.
        has_nans (bool): Indicates whether the input data contains NaN values.

    """

    def __init__(self, margin=1.0, has_nans=False):
        super().__init__()

        if not has_nans:
            self.loss = nn.TripletMarginLoss(margin=margin, p=2, reduction="mean")
            self.mse = nn.MSELoss(reduction="mean")
        else:
            self.loss = TripletNANLoss(margin=margin)
            self.mse = MSENANLoss()

    def forward(
        self,
        X=None,
        Xt=None,
        y_true=None,
        thetas: list[torch.tensor] = None,
        model=None,
        nans_mask=None,
        **kwargs
    ):
        """
        Compute the cyclic triplet loss.

        Args:
            X (torch.Tensor): Input tensor.
            Xt (torch.Tensor): Transformed input tensor.
            y_true (torch.Tensor): True labels tensor.
            thetas (list[torch.Tensor]): List of theta tensors.
            model: The CPAB transformer model.
            nans_mask: Mask for NaN values.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: Computed loss.

        """

        loss = 0.0

        n_classes = torch.unique(y_true).tolist()
        nk = len(n_classes)
        batchsize, channels, input_shape = Xt.shape

        for i in n_classes:
            Xt_k = Xt[y_true == i]
            X_k = X[y_true == i]
            mask_k = nans_mask[y_true == i]

            if nk > 1:
                neg_classes = n_classes[n_classes != i]
                i_neg = torch.tensor(np.random.choice(neg_classes, 1), device=X.device)
            else:
                print("Warning: only one class in batch")
                i_neg = i

            mask_sum = torch.logical_not(mask_k).sum(dim=0)
            mask_sum[mask_sum == 0.0] = 1.0
            X_mean = Xt_k.sum(dim=0) / mask_sum

            if nk > 1:
                mask_neg_k = nans_mask[y_true == i_neg]
                mask_neg_k = torch.logical_not(mask_neg_k).sum(dim=0)
                mask_neg_k[mask_neg_k == 0.0] = 1.0
                Xt_k_neg = Xt[y_true == i_neg]
                X_neg_mean = Xt_k_neg.sum(dim=0) / mask_neg_k

            n_samples = Xt_k.shape[0]
            X_mean_stack = X_mean.repeat(n_samples, 1, 1)
            

            theta_k = thetas[y_true == i]
            X_mean_stack = model.transform_data_inverse(X_mean_stack, theta_k)

            # ICAE 
            loss += self.mse(X_k, X_mean_stack)
            # ICAE-triplet
            if nk > 1:
                X_neg_mean_stack = X_neg_mean.repeat(n_samples, 1, 1)
                X_neg_mean_stack = model.transform_data_inverse(X_neg_mean_stack, theta_k)

                loss += self.loss(X_k, X_mean_stack, X_neg_mean_stack)

        return loss


class MSENANLoss(nn.Module):
    """
    Mean Squared Error (MSE) loss that handles NaN values by ignoring them.

    Args:
        reduction (str): Reduction type. Either "mean" or "none".

    """

    def __init__(self, reduction="mean"):
        super(MSENANLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, obs):
        mask1 = torch.logical_not(torch.isnan(pred))
        mask2 = torch.logical_not(torch.isnan(obs))
        mask = torch.logical_and(mask1, mask2)
        pred[~mask] = 0
        obs[~mask] = 0
        loss = (pred - obs) ** 2

        if self.reduction == "mean":
            loss = torch.mean(loss)

        return loss


class TripletNANLoss(nn.Module):
    """
    Triplet loss that handles NaN values by replacing them with zeros.

    Args:
        margin (float): Margin value for the triplet loss.

    """

    def __init__(self, margin=1):
        super(TripletNANLoss, self).__init__()
        self.margin = margin
        self.msenan = MSENANLoss(reduction="none")

    def forward(self, anchor, pos, neg):
        loss = torch.max(
            self.msenan(anchor, pos) - self.msenan(anchor, neg) + self.margin,
            torch.zeros_like(anchor),
        )

        return torch.mean(loss)
