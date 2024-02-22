"""
Author: Charlie
Purpose: Data loading tools
"""


import os
import numpy as np

from PIL import Image
from tqdm import tqdm


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
