"""
Author: Charlie
Purpose: General tools
"""


import os
import shutil


def create_folder(path, overwrite=False):
    """
    Creates an overwritable user specified folder

    Parameters:
    - path (str): path to folder
    - overwrite (bool): flag for remaking existing folder
    """

    if os.path.exists(path):
        if overwrite:
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)
