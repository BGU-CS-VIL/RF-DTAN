from loss.smoothness_prior import PriorLoss
from loss.alignment_loss import WCSS, WCSSTriplet, ICAE, ICAETriplet
from helper.util import align_dataset

from models.train_utils import (
    train_epoch,
    val_epoch,
    save_checkpoint,
)

from tqdm import tqdm
import os

import torch.optim as optim
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        exp_name: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        args,
    ) -> None:
        # args
        self.args = args
        self.exp_name = exp_name
        self.batch_size = args.batch_size
        self.n_epochs = args.n_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loader = train_loader
        self.val_loader = val_loader

        # DTAN
        self.model = model.to(self.device)
        self.T = self.model.T  # needs adjustments for multiscale

        # # training
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            eps=1e-8,
            weight_decay=0.0001,
            amsgrad=True,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            T_max=self.n_epochs, optimizer=self.optimizer, eta_min=0.00001
        )

        # best val
        self.best_chk = 0
        self.load_best = args.load_best
        # Losses
        self.WCSS_loss = args.WCSS_loss
        self.WCSS_triplets_loss = args.WCSS_triplets_loss
        self.ICAE_loss = args.ICAE_loss
        self.ICAE_triplets_loss = args.ICAE_triplets_loss
        self.smoothness_prior = args.smoothness_prior
        self.has_nans = torch.any(torch.isnan(train_loader.dataset[:][0]))

        self.loss_weights = args.loss_weights
        self.loss_funcs = {}

        # init losses
        if self.ICAE_loss:
            self.loss_funcs["ICAE_loss"] = ICAE(has_nans=self.has_nans)
        if self.ICAE_triplets_loss:
            self.loss_funcs["ICAE_triplets_loss"] = ICAETriplet(
                margin=1, has_nans=self.has_nans
            )
        if self.WCSS_loss:
            self.loss_funcs["WCSS_loss"] = WCSS()

        if self.WCSS_triplets_loss:
            self.loss_funcs["WCSS_triplets_loss"] = WCSSTriplet()

        if self.smoothness_prior:
            self.loss_funcs["prior_loss"] = PriorLoss(
                self.T, args.lambda_smooth, args.lambda_var
            )

        # init empty loss dict for plotting loss later
        self.train_loss = {k: [] for k in self.loss_funcs.keys()}
        self.train_loss["loss"] = []
        self.val_loss = {k: [] for k in self.loss_funcs.keys()}
        self.val_loss["loss"] = []

    def train(self, print_model=False):

        # Print model
        if print_model:
            print(self.model)
            pytorch_total_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            print("# parameters:", pytorch_total_params)

        min_loss = np.inf
        with tqdm(
            range(1, self.n_epochs + 1), position=0, leave=True, unit="Epoch"
        ) as tepoch:
            for epoch in tepoch:

                epoch_train_loss = train_epoch(
                    self.train_loader,
                    self.model,
                    self.device,
                    self.optimizer,
                    self.loss_funcs,
                    self.loss_weights,
                )
                if not self.args.no_validation:
                    epoch_val_loss = val_epoch(
                        self.val_loader,
                        self.model,
                        self.device,
                        self.loss_funcs,
                        self.loss_weights,
                    )
                else:
                    epoch_val_loss = epoch_train_loss

                if self.scheduler is not None:
                    self.scheduler.step()

                if not self.args.no_validation and (epoch_val_loss["loss"] < min_loss):
                    best_chk = epoch
                    min_loss = epoch_val_loss["loss"]

                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch_val_loss,
                        self.exp_name,
                        self.args.ckpt_dir,
                    )


                self.update_loss_dicts(epoch_train_loss, epoch_val_loss)
                tepoch.set_postfix(
                    train_loss=epoch_train_loss["loss"], val_loss=epoch_val_loss["loss"]
                )

        # Load best model based on validation loss
        if self.load_best:
            checkpoint_path = os.path.join(self.args.ckpt_dir, self.exp_name)

            checkpoint_path += "_checkpoint.pth"
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print("Loaded checkpoint from:", checkpoint_path, "epoch", best_chk)

        
        torch.cuda.empty_cache()
        self.model.eval()

    def update_loss_dicts(self, epoch_train_loss, epoch_val_loss):
        # with torch.no_grad():
        for key in epoch_train_loss.keys():
            self.train_loss[key].append(epoch_train_loss[key])
            self.val_loss[key].append(epoch_val_loss[key])