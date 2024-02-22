"""
Purpose: Clustering Methods
Author: Charlie
"""


import os
import pickle
import numpy as np

from tqdm import tqdm
from sklearn_extra.cluster import KMedoids

from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score, v_measure_score
from sklearn.metrics import homogeneity_score, completeness_score


def get_best_k_info(u_measures, all_data, dataset):

    bar_results, subset_results = {}, {}

    print("\n------ Get Best K Info ------\n")

    for t_key in tqdm(u_measures, desc="Gathering"):

        measures = u_measures[t_key]["mul_dbi_chs"]["max"]
        index = np.argmax(measures) + 2

        bar_results[t_key] = []
        subset_results[t_key] = {}

        for d_key in all_data[t_key].keys():

            d_results = {}
            d_results["name"] = d_key

            e_indices = all_data[t_key][d_key]["e_indices"][index]

            exemplars = []
            for e_idx in e_indices:
                exemplars.append(dataset.samples[e_idx])

            e_preds = all_data[t_key][d_key]["e_preds"][index]

            subset_samples, subset_labels, subset_preds = [], [], []

            for i in range(dataset.samples.shape[0]):
                if e_preds[i] != -1:
                    subset_samples.append(dataset.samples[i])
                    subset_labels.append(dataset.labels[i])
                    subset_preds.append(e_preds[i])

            subset_samples = np.asarray(subset_samples)
            subset_labels = np.asarray(subset_labels)
            subset_preds = np.asarray(subset_preds)

            ari = adjusted_rand_score(subset_labels, subset_preds)
            nmi = normalized_mutual_info_score(subset_labels, subset_preds)
            v_measure = v_measure_score(subset_labels, subset_preds)
            homogeneity = homogeneity_score(subset_labels, subset_preds)
            completeness = completeness_score(subset_labels, subset_preds)

            d_results["ari"] = ari
            d_results["nmi"] = nmi
            d_results["v_measure"] = v_measure
            d_results["homogeneity"] = homogeneity
            d_results["completeness"] = completeness

            bar_results[t_key].append(d_results)

            subset_results[t_key][d_key] = {"samples": subset_samples,
                                            "labels": subset_labels,
                                            "preds": subset_preds,
                                            "exemplars": exemplars}

    return bar_results, subset_results


def get_preds_from_exemplars(data, all_exemplars,
                             t_max=1.00, t_min=0.80, min_size=100):

    # Gather: Class Memberships

    memberships = np.zeros((data.shape[0], len(all_exemplars)))
    for i, exemplar in enumerate(all_exemplars):
        memberships[:, i] = 1 - exemplar.reshape(-1)

    memberships = (memberships - memberships.min()) / (memberships.max() - memberships.min())

    # Explore: Different Thresholds

    t = t_max
    keep_going = 1
    while keep_going:

        # - Gather class predictions

        predictions = []
        for confidences in memberships:
            if (confidences < t).all():
                predictions.append(-1)
            else:
                label = np.argmax(confidences)
                predictions.append(label)
        predictions = np.asarray(predictions)

        # - Gather number of samples per class

        counts = []
        for label in np.unique(predictions):
            if label == -1:
                continue
            counts.append(len(np.where(label == predictions)[0]))
        counts = np.asarray(counts)

        # - Finish or lower class acceptance threshold until "t_min"

        if (counts > min_size).all() or t < t_min:
            keep_going = 0
            if t < t_min:
                # Find where counts isn't > min_size and set deficit with -1's
                # predictions = np.zeros(predictions.shape[0])
                # indices = np.where(counts
                break
        else:

            t = t - 0.01

    return predictions, t


def run_k_clustering(data, k, title="Exploring K"):

    data = data / data.max()

    all_results = {"m_preds": [], "e_preds": [],
                   "thresholds": [], "e_indices": []}

    pbar = tqdm(total=k-2, desc="K-Medoids - %s" % title)

    all_models = []
    for i in range(2, k, 1):

        model = KMedoids(n_clusters=i, init="k-medoids++",
                         metric="precomputed").fit(data)

        m_preds = model.labels_

        all_results["m_preds"].append(m_preds)
        all_models.append(model)

        pbar.update(1)

    pbar.close()

    for i, model in enumerate(all_models):

        exemplars = np.asarray([data[j] for j in model.medoid_indices_])
        e_preds, thresh = get_preds_from_exemplars(data, exemplars)

        # if len(e_preds) == 0:
        #    break

        e_indices = []
        for e in exemplars:
            for i, row in enumerate(data):
                if (e == row).all():
                    e_indices.append(i)
                    break

        all_results["e_indices"].append(e_indices)
        all_results["e_preds"].append(e_preds)
        all_results["thresholds"].append(thresh)

    return all_results


def run_clustering(path, all_matrices, max_k, truths, create):

    path_file = os.path.join(path, "clustering.pkl")

    if create or not os.path.exists(path_file):

        print("\n------ Clustering Predictions ------\n")

        results = {}

        # Gather: Clustering Predictions

        for current_key in all_matrices.keys():

            if current_key != "features":
                continue

            results[current_key] = {}

            data = all_matrices[current_key]

            for dims in data.keys():
                matrix = data[dims]
                title = "%s (%s)" % (current_key, dims)
                k_results = run_k_clustering(matrix, max_k, title)
                results[current_key][dims] = k_results

        # Save: Clustering Results

        pickle.dump(results, open(path_file, "wb"))

    else:

        results = pickle.load(open(path_file, "rb"))

    return results
