"""
Author: Charlie
Purpose: Measure tools
"""


import numpy as np

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def euclidean(u, v):

    u = u.reshape(-1)
    v = v.reshape(-1)

    return np.linalg.norm(u - v)


def perceptual_dissim(u, v):

    psnr = psnr_dissim(u, v)
    ssim = ssim_dissim(u, v)

    return ssim + psnr


def psnr_dissim(u, v, min_val=0, max_val=50):

    # if len(u.shape) == 1:
    #     size = int(np.sqrt(len(u)))
    #     u = u.reshape(size, size)
    #     v = v.reshape(size, size)

    measure = psnr(u, v, data_range=u.max() - v.min())
    measure = (measure - min_val) / (max_val - min_val)

    return 1 - measure


def ssim_dissim(u, v):

    # if len(u.shape) == 1:
    #     size = int(np.sqrt(len(u)))
    #     u = u.reshape(size, size)
    #     v = v.reshape(size, size)

    measure = ssim(u, v, data_range=u.max() - v.min())

    if measure < 0:
        measure = 0

    return 1 - measure


def select_measure(choice):

    if choice == 0:
        measure = ssim_dissim
        name = "ssim"

    elif choice == 1:
        measure = euclidean
        name = "euclidean"

    elif choice == 2:
        measure = psnr_dissim
        name = "psnr"

    elif choice == 3:
        measure = perceptual_dissim
        name = "perceptual"

    else:
        raise NotImplementedError

    return measure, name
