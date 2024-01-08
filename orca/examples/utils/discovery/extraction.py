"""
Author: Charlie
Purpose: Image feature extraction methods for knowledge discovery
"""

import numpy as np

from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from utils.neural.models import load_trained_model


def get_tsne_features(data, num_features):

    # data = data.samples.reshape(data.samples.shape[0], -1)

    model = TSNE(n_components=num_features)

    return model.fit_transform(data)


def get_pca_features(data, num_features):

    # data = data.samples.reshape(data.samples.shape[0], -1)

    model = PCA(n_components=num_features)

    return model.fit_transform(data)


def get_nn_features(data, params):

    model = load_trained_model(params)
    loader = DataLoader(data, shuffle=False, batch_size=32,
                        num_workers=params["system"]["num_workers"])

    all_features = {"high": [], "low": [], "recon": [], "original": []}
    for samples, _ in tqdm(loader, desc="Extracting Features"):

        samples = samples.to(params["system"]["accelerator"])

        all_high, all_low, recons = model(samples)

        all_low = all_low.detach().cpu().numpy()
        all_high = all_high.detach().cpu().numpy()

        recons = recons.detach().cpu()
        samples = samples.detach().cpu()

        for high, low in zip(all_high, all_low):
            all_features["low"].append(low.reshape(-1))
            all_features["high"].append(high.reshape(-1))

        all_features["recon"].append(recons)
        all_features["original"].append(samples)

    all_features["low"] = np.asarray(all_features["low"])
    all_features["high"] = np.asarray(all_features["high"])

    return all_features
