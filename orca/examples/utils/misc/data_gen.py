"""
Author: Charlie
Purpose: Generate and save image datasets
"""


import os
import shutil
import numpy as np

from tqdm import tqdm
from PIL import Image
from torchvision import datasets


def create_folder(path):
    """
    Creates folder, removes existing data

    Parameters:
    - path (str): path to folder
    """

    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path)


def save_dataset(path, dataset, choice):
    """
    Saves dataset to disk, overwrites existing

    Parameters:
    - path (str): path to download or find dataset
    - (dict[str, np.ndarray[any]]): dataset information
    - choice (str): dataset parition, either 'train' or 'test'
    """

    desc = "Saving %s Data" % choice.capitalize()
    for label in tqdm(np.unique(dataset["labels"]), desc=desc):

        # Gather: Class Samples

        indices = np.where(label == dataset["labels"])
        class_samples = dataset["samples"][indices]

        # Create: Class Folder

        path_folder = os.path.join(path, str(label).zfill(4))
        create_folder(path_folder)

        # Save: Class Data

        for i, sample in enumerate(class_samples):
            path_file = os.path.join(path_folder, str(i).zfill(6) + ".png")
            Image.fromarray(sample).save(path_file)


def load_cifar10(path, choice):
    """
    Download or load existing CIFAR-10 dataset

    Parameters:
    - path (str): path to download or find dataset
    - choice (str): dataset parition, either 'train' or 'test'

    Returns:
    - (dict[str, np.ndarray[any]]): dataset information
    """

    if choice == "train":
        use_train = True
    else:
        use_train = False

    data = datasets.CIFAR10(root=path, train=use_train, download=True)

    return {"samples": data.data, "labels": np.asarray(data.targets)}


def generate_and_save(path, choice):
    """
    Generate and save image datasets

    Parameters:
    - path (str): root path for experiment operations
    - choice (str): dataset parition, either 'train' or 'test'
    """

    # Gather: Dataset
    # - Loads dataset from disk or downloads it

    path_save = os.path.join(path, "raw")
    data = load_cifar10(path_save, choice)

    # Save: Dataset

    path_save = os.path.join(path, choice)
    save_dataset(path_save, data, choice)


def experiment(path):
    """
    Generate and save image datasets

    Parameters:
    - path (str): root path for experiment operations
    """

    for choice in ["train", "test"]:
        generate_and_save(path, choice)


if __name__ == "__main__":
    """
    Run experiment
    """

    # path = "/develop/data/cifar"
    path = "/Users/slane/Downloads/cifar"

    experiment(path)
