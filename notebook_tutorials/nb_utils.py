
import numpy as np
import matplotlib.pyplot as plt


def plot_ts(X, y, n, X2=None, title="", plot_mean=False, savefig=False):
    
    
    # for univariate
    if len(X.shape) == 2:
        X = np.expand_dims(X,1)
    np.random.seed(0)
    channels = X.shape[1]
    classes = np.unique(y)
    nc = len(classes)
    cols = min(nc, 2) 
    rows = int(np.ceil(nc/cols))
    fig, axes = plt.subplots(rows, cols, sharey=True, dpi=100)
    fig.set_size_inches(12, 4*rows)
    axes = axes.ravel()
    for i, label in enumerate(classes):
        class_idx = y == label 
        X_in_class = X[class_idx]
        if X2 is not None:
            X2_k = X2[class_idx]
        class_idx = np.random.choice(len(X_in_class), n)
        X_in_class =X_in_class[class_idx]
        if plot_mean:
            # aligned mean
            X_mean = np.expand_dims(np.mean(X_in_class, axis=0), axis=0)

            if X2 is not None:
                # plot original data
                X_in_class = X2_k[class_idx]
            for c in range(channels):
                axes[i].plot(X_in_class[:,c,:].T, color='grey', alpha=0.5)
                axes[i].plot(X_mean[:,c,:].T, label='Mean', linewidth=3)

        else:
            for c in range(channels):
                axes[i].plot(X_in_class[:,c,:].T)
        axes[i].set_title(f"Class {label}", fontsize=18)
        axes[i].grid(True)
    
    i+=1
    while i < len(axes):
        axes[i].remove()
        i +=1
    
    if title:
        plt.suptitle(title)
    plt.tight_layout()