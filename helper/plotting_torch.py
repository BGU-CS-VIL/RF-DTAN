import numpy as np
import torch
from helper.UCR_loader import processed_UCR_data
#from tsai.data.external import get_UCR_data as get_UCR_data_tsai
from tslearn.datasets import UCR_UEA_datasets

from pathlib import Path

import matplotlib.pyplot as plt


def plot_mean_signal(
    X_aligned_within_class, X_within_class, ratio, class_num, dataset_name, savefig=False
):

    # check data dim
    if len(X_aligned_within_class.shape) < 3:
        X_aligned_within_class = np.expand_dims(X_aligned_within_class, axis=-1)

    nb, n_channels, signal_len = X_within_class.shape
    t = range(signal_len)
    # Compute mean signal and variance
    X_mean_t = np.nanmean(X_aligned_within_class, axis=0)
    X_std_t = np.nanstd(X_aligned_within_class, axis=0)
    upper_t = X_mean_t + X_std_t
    lower_t = X_mean_t - X_std_t

    X_mean = np.nanmean(X_within_class, axis=0)
    X_std = np.nanstd(X_within_class, axis=0)
    upper = X_mean + X_std
    lower = X_mean - X_std

    # set figure size
    [w, h] = ratio  # width, height
    plt.style.use("seaborn-darkgrid")
    f = plt.figure(1)
    f.set_size_inches(w, n_channels * h)

    title_font = 18
    rows = 2
    cols = 2
    # plot each channel
    # list of 10 colors for multichannel
    colors = ["b", "c", "g", "r", "m", "y", "k", "orange", "purple", "brown"]
    for channel in range(n_channels):
        plot_idx = 1
        # Misaligned Signals
        if channel == 0:
            ax1 = f.add_subplot(rows, cols, plot_idx)
        if n_channels == 1:
            ax1.plot(X_within_class[:, channel, :].T)
        else:
            ax1.plot(X_within_class[:, channel, :].T, color=colors[channel], alpha=0.4)
        plt.tight_layout()
        plt.xlim(0, signal_len)

        if n_channels == 1:
            # plt.title("%d random test samples" % (N))
            plt.title("Misaligned signals", fontsize=title_font)
        else:
            plt.title(f"Channel: {channel}, Train data mean signal samples)")
        plot_idx += 1

        # Misaligned Mean
        ax2 = f.add_subplot(rows, cols, plot_idx)
        ax2.plot(t, X_mean[channel], "r", label="Average signal")
        ax2.fill_between(
            t,
            upper[channel],
            lower[channel],
            color="r",
            alpha=0.2,
            label=r"$\pm\sigma$",
        )
        plt.xlim(0, signal_len)

        if n_channels == 1:
            plt.title("Misaligned average signal", fontsize=title_font)
        else:
            plt.title(f"Channel: {channel}, Test data mean signal samples)")

        plot_idx += 1

        # Aligned signals
        if channel == 0:
            ax3 = f.add_subplot(rows, cols, plot_idx)
        
        if n_channels == 1:
            ax3.plot(X_aligned_within_class[:, channel, :].T)
        else:
            ax3.plot(X_aligned_within_class[:, channel, :].T, color=colors[channel], alpha=0.4)
        plt.title("Aligned signals", fontsize=title_font)
        plt.xlim(0, signal_len)

        plot_idx += 1

        # Aligned Mean
        ax4 = f.add_subplot(rows, cols, plot_idx)
        # plot transformed signal
        t = np.arange(X_mean_t.shape[-1])
        ax4.plot(X_mean_t[channel, :], label="Average signal")
        ax4.fill_between(
            t,
            upper_t[channel],
            lower_t[channel],
            color="#539caf",
            alpha=0.6,
            label=r"$\pm\sigma$",
        )

        # plt.legend(loc='upper right', fontsize=12, frameon=True)
        plt.title("Aligned average signal", fontsize=title_font)
        plt.xlim(0, signal_len)
        plt.tight_layout()

        plot_idx += 1


    plt.suptitle(f"{dataset_name}: class-{class_num}", fontsize=title_font + 2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if savefig:
        Path(f"./figures/{dataset_name}").mkdir(parents=True, exist_ok=True)
        f.savefig(
            f"./figures/{dataset_name}/{dataset_name}: class-{class_num}.pdf", dpi=200
        )



def plot_signals(model, device, dataset_name, sets_dict, N=10, seed=0, savefig=False):
    # Close any remaining plots
    plt.close("all")
    torch.manual_seed(seed)
    np.random.seed(seed)
    with torch.no_grad():
        UCR_data = UCR_UEA_datasets()
        # tslearn format (N, sz, C). Torch needs (N, C, sz)
        # X_train, y_train, X_test, y_test = UCR_data.load_dataset(dataset_name)

        # X_train, y_train, X_test, y_test = processed_UCR_data(
        #     X_train, y_train, X_test, y_test, replace_nans=False, swap_channel_dim=True
        # )
        
        set_names = ["train", "test"]

        with torch.no_grad():
            for set_name in set_names:
                # torch dim
                X, y = sets_dict[set_name]
                classes = np.unique(y).tolist()
                for label in classes:
                    class_idx_bool = y == label
                    Xk = X[class_idx_bool]
                    sample_idx = np.random.choice(len(Xk), N)  # N samples

                    Xk = torch.Tensor(Xk[sample_idx]).to(device)
                    outputs = model(Xk, return_nans=True)

                    Xk = Xk.cpu().numpy()
                    Xk_aligned = outputs["data"].cpu().numpy()

                    plot_mean_signal(
                        Xk_aligned,
                        Xk,
                        ratio=[10, 6],
                        class_num=label,
                        dataset_name=f"{dataset_name}-{set_name}",
                        savefig=savefig,
                    )
                    plt.show()
