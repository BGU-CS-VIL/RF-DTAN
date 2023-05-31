# tslearn
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

# sklearn
from sklearn.metrics import accuracy_score
import torch
from helper.util import align_dataset
import numpy as np


def get_means(X, y):
    # vars and placeholders
    N, channels, input_shape = X.shape
    n_classes = len(np.unique(y))
    class_names = np.unique(y, axis=0)

    aligned_means = np.zeros((n_classes, channels, input_shape))  # currently univariate
    ncc_labels = []

    # Train set within class Euclidean mean
    for i, class_num in enumerate(class_names):
        train_class_idx = y == class_num  # get indices
        X_train_aligned_within_class = X[train_class_idx]
        aligned_means[i, :, :] = np.nanmean(X_train_aligned_within_class, axis=0)
        ncc_labels.append(class_num)

    ncc_labels = np.asarray(ncc_labels)

    return aligned_means, ncc_labels


def NearestCentroidClassification(
    X_train, X_test, y_train_n, y_test_n, dataset_name, metric="euclidean"
):
    """

    :param X_train: if using DTAN, should already be aligned
    :param X_test: if using DTAN, should already be aligned
    :param y_train_n: numerical labels (not one-hot)
    :param y_test_n: numerical labels (not one-hot)
    :param dataset_name:
    :return: test set NCC accuracy
    """

    # vars and placeholders
    N, channels, input_shape = X_train.shape
    n_classes = len(np.unique(y_train_n))
    class_names = np.unique(y_train_n, axis=0)

    aligned_means = np.zeros((n_classes, channels, input_shape))
    ncc_labels = []

    # Train set within class Euclidean mean
    for i, class_num in enumerate(class_names):
        train_class_idx = y_train_n == class_num  # get indices
        X_train_aligned_within_class = X_train[train_class_idx]
        aligned_means[i, :, :] = np.nanmean(X_train_aligned_within_class, axis=0)
        ncc_labels.append(class_num)

    ncc_labels = np.asarray(ncc_labels)

    # Nearest neighbor classification - using euclidean distance
    knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric=metric)
    knn_clf.fit(aligned_means, ncc_labels)

    predicted_labels = knn_clf.predict(X_test)
    acc = accuracy_score(y_test_n, predicted_labels)

    return acc


def NCC_with_means(X_means, y_means, X_test, y_test_n, metric="euclidean"):
    """

    :param X_train: if using DTAN, should already be aligned
    :param X_test: if using DTAN, should already be aligned
    :param y_train_n: numerical labels (not one-hot)
    :param y_test_n: numerical labels (not one-hot)
    :param dataset_name:
    :return: test set NCC accuracy
    """

    # Nearest neighbor classification - using euclidean distance
    knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric=metric)
    knn_clf.fit(X_means, y_means)

    predicted_labels = knn_clf.predict(X_test)
    acc = accuracy_score(y_test_n, predicted_labels)

    return acc


def NCC_pipeline(model, train_loader, test_loader, args, run=None, verbose=True):
    # NCC
    res_dict = {}
    with torch.no_grad():
        model.eval()
        X_train_aligned, _, y_train1, _ = align_dataset(train_loader, model, device="cuda")
        X_test_aligned, _, y_test1, _ = align_dataset(test_loader, model, device="cuda")
        X_train_aligned = X_train_aligned.detach().cpu().numpy()
        X_test_aligned = X_test_aligned.detach().cpu().numpy()
        y_train1 = y_train1.detach().cpu().numpy()
        y_test1 = y_test1.detach().cpu().numpy()

        torch.cuda.empty_cache()

    X_means, y_means = get_means(X_train_aligned, y_train1)
    X_train, y_train = train_loader.dataset[:]
    X_train, y_train = X_train.detach().cpu().numpy(), y_train.detach().cpu()
    X_test, y_test = test_loader.dataset[:]
    X_test, y_test = X_test.detach().cpu().numpy(), y_test.detach().cpu()
    # variance
    k_classes = np.unique(y_test)
    n_classes = len(k_classes)
    wcss = 0

    for k in k_classes:
        Xk_aligned = X_test_aligned[y_test == k]
        Xk = X_test[y_test == k]
        X_mean = np.expand_dims(Xk_aligned.mean(axis=0), axis=0)
        wcss += Xk_aligned.var() / n_classes


    res_dict[f"variance"] = wcss

    # clean some memory
    del train_loader

    baseline_acc = NearestCentroidClassification(
        X_train,
        X_test,
        y_train,
        y_test,
        args.dataset,
        metric="euclidean",
    )

    dtan_acc = NCC_with_means(
        X_means,
        y_means,
        X_test_aligned,
        y_test1,
        metric="euclidean",
    )

    res_dict[f"Baseline"] = baseline_acc
    res_dict[f"DTAN"] = dtan_acc


    if verbose:
        print("NCC Pipeline")
        print(f"Baseline: {args.dataset} results: {baseline_acc}")
        print(f"DTAN: {args.dataset} results: {dtan_acc}")

    return res_dict


