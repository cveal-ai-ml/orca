"""
Purpose: Test using pre-trained model predictions for knowledge discovery
"""


import os
import sys
import torch
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from torchvision import models
from scipy.stats import wasserstein_distance
from torchmetrics.regression import KLDivergence
from torchmetrics.functional.pairwise import pairwise_cosine_similarity


class Network(torch.nn.Module):

    def __init__(self):

        super().__init__()

        if sys.platform == "darwin":
            device = "mps"
        else:
            device = "cuda"

        self.device = device
        self.weights = models.ResNet50_Weights.DEFAULT
        self.arch = models.resnet50(weights=self.weights)
        self.arch = self.arch.eval()
        self.arch = self.arch.to(self.device)

    def forward(self, x):

        return self.arch(x)

    def test(self, dataset):

        dataset = torch.utils.data.DataLoader(dataset, shuffle=False,
                                              batch_size=256)

        all_preds = []
        for sample, _ in tqdm(dataset, desc="Processing"):
            sample = sample.to(self.device)

            with torch.no_grad():
                preds = self(sample)
                preds = preds.softmax(dim=1)

            all_preds.append(preds.cpu())

        all_preds = torch.vstack(all_preds)

        return all_preds.numpy()


class Dataset:

    def __init__(self, samples, labels, transforms):

        self.transforms = transforms
        self.samples = samples
        self.labels = labels

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        sample = Image.open(self.samples[index]).convert("RGB")
        sample = self.transforms(sample)

        return sample, self.labels[index]


def plot_matrix(path_file, data, title, fig_size=(14, 8), font_size=14):

    fig, ax = plt.subplots(figsize=fig_size)

    ax.imshow(data, cmap="gray")

    ax.set_title(title, fontsize=font_size)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(path_file)


def plot_bars(path_file, data, width=0.01, fig_size=(14, 8), font_size=14):

    fig, ax = plt.subplots(figsize=fig_size)

    ax.bar(np.arange(len(data)), data, width=width,
           edgecolor="black", color="darkblue")

    tag = int(path_file.split("/")[-1].strip(".png").strip("class_"))
    title = "Class Confidences - Truth Label Class %s" % tag

    ax.set_title(title, fontsize=font_size)
    ax.set_xlabel("Classes", fontsize=font_size)
    ax.set_ylabel("Frequency", fontsize=font_size)

    fig.tight_layout()

    fig.savefig(path_file)


def ivat(matrix, verbose=1):
    """
    Calculates improved Visual Assessment Tendency (iVAT)

    Parameters:
    - matrix (np.ndarray[float]): VAT matrix
    - verbose (int): flag to show progress bar

    Returns:
    - (np.ndarray[float]): iVAT matrix
    """

    ivat_matrix = np.zeros_like(matrix)

    if verbose:
        pbar = tqdm(total=matrix.shape[0], desc="Calculating iVAT")

    for r in range(1, matrix.shape[0]):

        j = np.argmin(matrix[r, :r])
        ivat_matrix[r, j] = matrix[r, j]

        for c in range(r):
            ivat_matrix[r, c] = max(matrix[r, j], ivat_matrix[j, c])

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ivat_matrix[i, j] = ivat_matrix[j, i]

        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()

    return ivat_matrix


def vat(matrix, verbose=1):
    """
    Calculates Visual Assessment Tendency (VAT)

    Parameters:
    - matrix (np.ndarray[float]): dissimilarity measure matrix
    - verbose (int): flag to show progress bar

    Returns:
    - (tuple[np.ndarray[float], list[int]): VAT matrix & sorted indices
    """

    assert len(matrix.shape) == 2, "VAT must be a one dimensional matrix"
    assert matrix.shape[0] == matrix.shape[1], "VAT shape must be squared"

    # Select: Origin Node

    i = np.argmax(matrix)

    i, j = np.unravel_index(i, matrix.shape)

    # Define: Sets I, J

    all_i = []
    all_j = list(range(matrix.shape[0]))

    # Update: Sets I, J

    all_i.append(i)
    all_j.remove(i)

    # Calculate: Ordered Dissimilarity Image (ODI)

    p = np.zeros(matrix.shape[0]).reshape(-1).astype(int)
    p[0] = i

    if verbose:
        pbar = tqdm(total=matrix.shape[0], desc="Calculating VAT")

    for z in range(1, matrix.shape[0]):

        all_values, indices = [], []
        for i in all_i:
            for j in all_j:
                indices.append([i, j])
                all_values.append(matrix[i, j])

        all_values = np.asarray(all_values)

        index = np.argmin(all_values)

        i, j = indices[index]

        p[z] = j
        all_i = all_i + [j]
        all_j.remove(j)

        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()

    # Finalize: ODI

    vat_matrix = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            vat_matrix[i, j] = matrix[p[i], p[j]]

    return (vat_matrix, all_i)


def euclidean_distance(u, v):

    return np.sqrt(np.sum((u - v) ** 2))


def hellinger_distance(u, v):

    return 1 / np.sqrt(2) * euclidean_distance(np.sqrt(u), np.sqrt(v))


def kl_divergence(u, v):

    u = torch.tensor(u).view(1, -1)
    v = torch.tensor(v).view(1, -1)

    measure = KLDivergence()

    return float(measure(u, v))


def cosine_similarity(u, v):

    u = torch.tensor(u).view(1, -1)
    v = torch.tensor(v).view(1, -1)

    return float(1 - pairwise_cosine_similarity(u, v))


def earth_movers_distance(u, v):

    return wasserstein_distance(u, v)


def make_comparisons(path, predictions, truth_labels):

    # Compare: Best Predictions

    path_folder = os.path.join(path, "comparisons", "labels")
    create_folder(path_folder)

    # - Create bar plots of prediction class label frequency

    for current_label in np.unique(truth_labels):
        indices = np.where(current_label == truth_labels)
        class_preds = np.asarray([np.argmax(pred) for pred in predictions[indices]])

        class_freqs = []
        for i in range(predictions.shape[-1]):
            indices = np.where(class_preds == i)
            class_freqs.append(len(indices[0]))

        class_preds = class_freqs

        name = "class_" + str(current_label).zfill(4) + ".png"
        path_file = os.path.join(path_folder, name)

        print("Saving: %s" % path_file)

        plot_bars(path_file, class_preds)

    # - Comparison matrices using prediction class labels as crisp measure

    indices = np.arange(len(predictions))
    np.random.shuffle(indices)

    # shuffled = np.asarray([np.argmax(ele) for ele in predictions[indices]])
    preds = np.asarray([np.argmax(ele) for ele in predictions])

    num_samples = len(truth_labels)

    matrix = np.zeros((num_samples, num_samples))

    pbar = tqdm(total=num_samples ** 2, desc="Creating Matrix")

    for i in range(num_samples):
        for j in range(num_samples):
            matrix[i][j] = int(preds[i] == preds[j])
            pbar.update(1)

    pbar.close()

    matrix = 1 - matrix
    # vat_matrix, _ = vat(matrix)
    # ivat_matrix, _ = ivat(vat_matrix)

    path_file = os.path.join(path_folder, "pred_label.png")
    plot_matrix(path_file, matrix, "Original")

    # path_file = os.path.join(path_folder, "vat.png")
    # plot_matrix(path_file, vat_matrix, "VAT")

    # path_file = os.path.join(path_folder, "ivat.png")
    # plot_matrix(path_file, ivat_matrix, "VAT")

    # Compare: Distributions

    path_folder_orig = os.path.join(path, "comparisons",
                                    "distributions", "original")

    path_folder_thresh = os.path.join(path, "comparisons",
                                      "distributions", "thershold")

    create_folder(path_folder_orig)
    create_folder(path_folder_thresh)

    # euclidean = {"matrix": np.zeros((num_samples, num_samples)),
    #              "function": euclidean_distance}

    # hellinger = {"matrix": np.zeros((num_samples, num_samples)),
    #              "function": hellinger_distance}

    kl = {"matrix": np.zeros((num_samples, num_samples)),
          "function": kl_divergence}

    cosine = {"matrix": np.zeros((num_samples, num_samples)),
              "function": cosine_similarity}

    # emd = {"matrix": np.zeros((num_samples, num_samples)),
    #        "function": earth_movers_distance}

    # all_measures = {"euclidean": euclidean, "hellinger": hellinger,
    #                 "kl": kl, "cosine": cosine, "emd": emd}

    all_measures = {"kl": kl, "cosine": cosine}

    preds = predictions.copy()
    # indices = np.where(predictions < 0.2)
    # preds[indices] = 0


    pbar = tqdm(total=num_samples ** 2, desc="Creating Matrices")

    for i in range(num_samples):
        for j in range(num_samples):
            for current_key in all_measures.keys():
                measure = all_measures[current_key]
                function = measure["function"]
                measure["matrix"][i][j] = function(preds[i], preds[j])

            pbar.update(1)

    pbar.close()

    for name in all_measures.keys():
        matrix = all_measures[name]["matrix"]
        path_file = os.path.join(path_folder_orig, "%s.png" % name)
        plot_matrix(path_file, matrix, "%s" % name.capitalize())

        t = 0.90
        matrix = 1 - (matrix / np.max(matrix))
        indices_pos = np.where(matrix >= t)
        indices_neg = np.where(matrix < t)

        matrix[indices_pos] = 1
        matrix[indices_neg] = 0

        path_file = os.path.join(path_folder_thresh, "%s.png" % name)
        plot_matrix(path_file, matrix, "%s" % name.capitalize())


def create_folder(path):

    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path)


def gather_data(path, transforms, percent):

    all_folders = os.listdir(path)
    all_folders.sort()

    pbar = tqdm(total=len(all_folders), desc="Loading Data")

    all_samples, all_labels = [], []
    for i, folder in enumerate(all_folders):
        folder = os.path.join(path, folder)

        all_files = os.listdir(folder)
        all_files.sort()

        num_samples = int(len(all_files) * (percent / 100))

        all_files = all_files[:num_samples]

        for file in all_files:
            all_samples.append(os.path.join(folder, file))
            all_labels.append(i)

        pbar.update(1)

    pbar.close()

    all_samples = np.asarray(all_samples)
    all_labels = np.asarray(all_labels)

    return Dataset(all_samples, all_labels, transforms)


def experiment(params):

    # Load: Model

    model = Network()

    # Load: Dataset

    transforms = model.weights.transforms()
    dataset = gather_data(params["paths"]["valid"], transforms,
                          params["dataset"]["percent_class_samples"])

    # Gather: Predictions

    path_file = os.path.join(params["paths"]["results"], "preds.pkl")

    if params["preds"]["create"] or not os.path.exists(path_file):
        create_folder(params["paths"]["results"])
        predictions = model.test(dataset)
        pickle.dump(predictions, open(path_file, "wb"))
    else:
        predictions = pickle.load(open(path_file, "rb"))

    # Perform: Comparison Analysis

    make_comparisons(params["paths"]["results"], predictions, dataset.labels)


if __name__ == "__main__":

    params = {}

    # params["paths"] = {"valid": "/develop/data/cifar/test",
    #                    "results": "/develop/results/new_test"}

    params["dataset"] = {"percent_class_samples": 25}

    params["paths"] = {"valid": "/Users/slane/Documents/research/data/cifar/test",
                       "results": "/Users/slane/Documents/research/results/new_test"}

    params["preds"] = {"create": 1}

    experiment(params)
