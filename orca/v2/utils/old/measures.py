"""
Author: Charlie
Purpose: Measure tools
"""


import torch
import numpy as np

from scipy.stats import wasserstein_distance
from torchmetrics.regression import KLDivergence
from sklearn.metrics import euclidean_distances as l2_norm
from sklearn.metrics.pairwise import cosine_similarity as cos_sim


def kl_divergence(u, v):

    v = torch.tensor(v)
    u_n = torch.zeros_like(v)
    u_n[:, :] = torch.tensor(u)

    measure = KLDivergence(reduction=None)

    return measure(u_n, v).numpy()


def cosine_similarity(u, v=None):

    if v is None:
        results = cos_sim(u)
    else:
        results = cos_sim(u, v)

    return np.round(1 - results, 3)


def euclidean_distance(u, v=None):

    if v is None:
        results = l2_norm(u)
    else:
        results = l2_norm(u, v)

    return results


def hellinger_distance(u):

    return 1 / np.sqrt(2) * euclidean_distance(np.sqrt(u))

# def cosine_similarity(u, v):
#
#     u = torch.tensor(u).view(1, -1)
#     v = torch.tensor(v).view(1, -1)
#
#     return float(1 - pairwise_cosine_similarity(u, v))


def earth_movers_distance(u, v):

    return wasserstein_distance(u, v)


def compare_neighbors(all_u, all_v):

    u_count = []
    for u in all_u:
        count = len(np.where(u == all_u)[0])
        u_count.append(count)

    results = []
    for i, u in enumerate(all_u):
        if u in all_v:
            count = len(np.where(u == all_v)[0])
            ratio = count / u_count[i]
        else:
            ratio = 0
        results.append(ratio)

    return 1 - np.mean(results)
