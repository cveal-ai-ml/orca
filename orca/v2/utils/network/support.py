"""
Purpose: Network Support Tools
Author: Charlie
"""


import os
import umap
import pickle
import numpy as np

from tqdm import tqdm
# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE

from utils.general import create_folder


def get_tsne_features(data, num_features):

    if num_features < 4:
        method = "barnes_hut"
    else:
        method = "exact"

    model = TSNE(n_jobs=4, n_components=num_features, method=method)

    return model.fit_transform(data)


def get_umap_features(data, num_features):

    model = umap.UMAP(n_components=num_features)

    return model.fit_transform(data)


def get_crisp(preds):

    indices = preds.argmax(axis=1)

    preds = np.zeros_like(preds)
    for i in range(preds.shape[0]):
        preds[i, indices[i]] = 1

    return preds


def top_k_to_crisp(preds):

    results = np.zeros_like(preds)
    indices = np.where(preds > 0)
    results[indices] = 1

    return results


def get_top_k(preds, k):

    results = np.zeros_like(preds)

    name = "Top K = %s" % k
    pbar = tqdm(total=len(preds), desc=name)

    for i, confidences in enumerate(preds):
        indices = np.argsort(confidences)[::-1][:k]
        for j in indices:
            results[i, j] = confidences[j]
        pbar.update(1)

    pbar.close()

    return results


def get_nn_predictions(path, model, dataset,
                       all_top_k, tsne_dims, create):

    path_file = os.path.join(path, "preds.pkl")

    if create or not os.path.exists(path_file):

        create_folder(path, overwrite=True)

        print("\n------ Network Predictions ------\n")

        results = model.test(dataset)

        print("\n------ TSNE Embeddings------\n")

        # Gather: All Embeddings
        # - Use TSNE to reduce to different feature dimensions

        features = results["features"]
        reduced_features = {}
        reduced_features[features.shape[1]] = features

        for f_size in tqdm(tsne_dims, desc="TSNE"):
            reduced = get_tsne_features(features, f_size)
            # reduced = get_umap_features(features, f_size)

            reduced_features[f_size] = reduced

        # # Gather: All Confidences

        # predictions = results["confidences"]
        # crisp_predictions = get_crisp(predictions)

        # soft, crisp = {}, {}
        # soft[predictions.shape[1]] = predictions
        # crisp[predictions.shape[1]] = crisp_predictions

        # print("\n------ Top K Confidences ------\n")

        # for top_k in all_top_k:

        #     soft_preds = get_top_k(predictions, top_k)
        #     crisp_preds = top_k_to_crisp(soft_preds)

        #     soft[top_k] = soft_preds
        #     crisp[top_k] = crisp_preds

        # all_results = {"preds_soft": soft, "preds_crisp": crisp,
        #                "features": reduced_features}

        all_results = {"features": reduced_features}

        pickle.dump(all_results, open(path_file, "wb"))

    else:

        all_results = pickle.load(open(path_file, "rb"))

    return all_results
