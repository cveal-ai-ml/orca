"""
Purpose: Test using pre-trained model predictions for knowledge discovery
"""


import os
import torch
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from torchvision import models


class Network(torch.nn.Module):

    def __init__(self, device="cuda"):

        super().__init__()

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


def plot_bars(path_file, data, width=0.01, fig_size=(14, 8), font_size=14):

    fig, ax = plt.subplots()

    all_names = ["class %s" % i for i in range(len(data))]

    all_positions = []

    for i in range(len(data)):

        if i == 0:
            position = np.arange(len(data))
        else:
            position = [ele + width for ele in all_positions[i - 1]]

        all_positions.append(position)

    fig, ax = plt.subplots(figsize=fig_size)

    color_range = np.arange(len(data)) + 1
    colors = plt.cm.turbo(color_range / float(max(color_range)))
    for i, (position, group) in enumerate(zip(all_positions, data)):
        ax.bar(position, group, width=width,
               edgecolor="black", color=colors[i])

    ax.set_xlabel("Models", fontsize=font_size)
    ax.set_ylabel("Performance", fontsize=font_size)

    ax.set_ylim([0, 1])
    ax.set_xticks([r + width * 5 for r in range(len(data[0]))],
                  all_names, fontsize=font_size)

    fig.tight_layout()

    fig.save(path_file)


def make_comparisons(path, predictions, truth_labels):

    # Compare: Best Predictions

    path_folder = os.path.join(path, "comparisons", "labels")
    create_folder(path_folder)

    for current_label in truth_labels:
        indices = np.where(current_label == truth_labels)
        class_preds = np.asarray([np.argmax(pred) for pred in predictions[indices]])

        name = "class_" + str(current_label).zfill(4) + ".png"
        path_file = os.path.join(path_folder, name)
        plot_bars(path_file, class_preds)

    # Compare: Distributions

    path_folder = os.path.join(path, "comparisons", "distributions")
    create_folder(path_folder)


def create_folder(path):

    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path)


def gather_data(path, transforms):

    all_folders = os.listdir(path)
    all_folders.sort()

    pbar = tqdm(total=len(all_folders), desc="Loading Data")

    all_samples, all_labels = [], []
    for i, folder in enumerate(all_folders):
        folder = os.path.join(path, folder)

        all_files = os.listdir(folder)
        all_files.sort()

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
    dataset = gather_data(params["paths"]["valid"], transforms)

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
    params["paths"] = {"valid": "/develop/data/cifar/test",
                       "results": "/develop/results/new_test"}

    params["preds"] = {"create": 0}

    experiment(params)
