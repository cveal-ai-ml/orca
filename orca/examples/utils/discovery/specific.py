"""
Author: Charlie
Purpose: Specific tools
"""


import os
import numpy as np

from PIL import Image
from tqdm import tqdm

from utils.general import create_folder


def load_image(path, img_flag):
    """
    Load target image from filesystem

    Parameters:
    - path (str): path to image
    - img_flag (str): PIL image format ("L" or "RGB")

    Returns:
    - (np.ndarray[uint8]): target image
    """

    return np.asarray(Image.open(path).convert(img_flag))


def save_results(path, dataset, preds):

    pbar = tqdm(total=len(dataset), desc="Saving Data")

    # Create: All Save Folders

    all_labels = list(np.unique(dataset.labels)) + [-1]
    for current_class in all_labels:
        path_save = os.path.join(path, str(current_class).zfill(3))
        create_folder(path_save)

    # Save: Clustered Samples

    for i in range(len(dataset)):
        folder = str(preds[i]).zfill(3)
        filename = str(i).zfill(6) + ".png"
        path_save = os.path.join(path, folder, filename)
        Image.fromarray(dataset.samples[i]).save(path_save)
        pbar.update(1)

    pbar.close()


def save_neighbors(path, all_results):

    for i, results in enumerate(all_results):
        path_folder = os.path.join(path, str(i).zfill(6))
        create_folder(path_folder)

        target, group = results["target"], results["group"]

        path_file = os.path.join(path_folder, "orig.png")
        Image.fromarray(target).save(path_file)

        for j, image in enumerate(group):
            path_file = os.path.join(path_folder, str(j).zfill(6) + ".png")
            Image.fromarray(image).save(path_file)
