"""
Author: Charlie
Purpose: Data loading tools
"""


import os
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader

from utils.neural.transforms import load_data_transforms


class Dataset:
    """
    Create a dataset template
    """

    def __init__(self, samples, labels, transforms):
        """
        Assign instance parameters

        Parameters:
        - samples (np.ndarray[float]): dataset samples
        - labels (np.ndarray[float]): dataset labels
        - transforms (np.ndarray[float]): dataset labels
        """

        self.samples = samples
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        """
        Return the number of dataset samples
        """

        return len(self.samples)

    def augmentations(self, filename):
        """
        Perform image augmentations data

        Parameters:
        - filename (str): image file

        Returns:
        - (torch.tensor[float]): post augmentation image
        """

        image = np.asarray(Image.open(filename).convert("RGB"))
        image = (image / 255.0).astype(np.float32)
        image = image.transpose((2, 0, 1))

        # return self.transforms(image=image)["image"].float()

        return image

    def __getitem__(self, index):
        """
        Return sample, label at current index

        Parameters:
        - index (int): current iteration counter

        Returns:
        - (tuple[any]): current indexed sample, label
        """

        image = self.augmentations(self.samples[index])
        label = self.labels[index]

        return (image, label)


def filter_dataset(dataset, percent):
    """
    Filter dataset into percent number of samples per class

    Parameters:
    - dataset (Dataset): dataset template
    - percent (float): percent of samples per class

    Returns:
    - (Dataset): Filtered dataset
    """

    all_samples, all_labels = [], []

    for current_class in np.unique(dataset.labels):

        indices = np.where(dataset.labels == current_class)
        class_samples = dataset.samples[indices]
        class_labels = dataset.labels[indices]

        num_samples = int(class_samples.shape[0] * (percent / 100))

        all_samples.append(class_samples[:num_samples])
        all_labels.append(class_labels[:num_samples])

    all_samples = np.vstack(all_samples)
    all_labels = np.hstack(all_labels)

    return Dataset(all_samples, all_labels, dataset.transforms)


def gather_data(path, transforms=None, file_types=[".png", ".jpg"]):
    """
    Gather dataset image files

    Parameters:
    - path (str): path to image dataset folder
    - transforms (np.ndarray[float]): dataset labels
    - file_types (list[str]): valid file types

    Returns:
    - (Dataset): dataset template
    """

    # Gather: All Dataset Files (Folders)

    all_folders = os.listdir(path)
    all_folders.sort()

    # Load: Supervised Learning Dataset

    all_samples, all_labels = [], []
    for i, current_folder in enumerate(all_folders):

        all_files = os.listdir(os.path.join(path, current_folder))
        all_files.sort()

        for current_file in all_files:
            path_file = os.path.join(path, current_folder, current_file)
            all_samples.append(path_file)
            all_labels.append(i)

    return Dataset(all_samples, all_labels, transforms)


def load_datasets(params):
    """
    Load training and validation datasets for DL training

    Parameters:
    - path_train (str): path to training dataset folder
    - path_valid (str): path to validation dataset folder

    Returns:
    - (dict[str, Dataset]): training and validation datasets
    """

    transforms = load_data_transforms(params["datasets"]["transforms"],
                                      params["datasets"]["interpolate"],
                                      params["datasets"]["sample_size"])

    train = gather_data(params["paths"]["train"], transforms["train"])
    valid = gather_data(params["paths"]["valid"], transforms["valid"])

    batch_size = params["network"]["batch_size"]
    num_workers = params["datasets"]["num_workers"]

    train = DataLoader(train, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, persistent_workers=True)

    valid = DataLoader(valid, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, persistent_workers=True)

    return {"train": train, "valid": valid}
