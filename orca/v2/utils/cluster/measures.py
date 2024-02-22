"""
Purpose: Measure Tools
Author: Charlie
"""


import numpy as np

from tqdm import tqdm

from scipy.stats import wasserstein_distance

from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import hamming_loss as h_distance
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import euclidean_distances as l2_norm
from sklearn.metrics import adjusted_rand_score, v_measure_score
from sklearn.metrics import homogeneity_score, completeness_score
from sklearn.metrics.pairwise import cosine_similarity as cos_sim


def get_supervised_measures(all_data, labels):

    ignore = ["max", "min", "avg"]

    all_results = {"ari": {}, "nmi": {}, "homogeneity": {},
                   "completeness": {}, "v_measure": {}}

    for current_key in all_data.keys():

        if current_key in ignore:
            continue

        data = all_data[current_key]["m_preds"]

        all_results["ari"][current_key] = []
        all_results["nmi"][current_key] = []
        all_results["v_measure"][current_key] = []
        all_results["homogeneity"][current_key] = []
        all_results["completeness"][current_key] = []

        for k_preds in data:

            ari = adjusted_rand_score(labels, k_preds)
            nmi = normalized_mutual_info_score(labels, k_preds)

            v_measure = v_measure_score(labels, k_preds)
            homogeneity = homogeneity_score(labels, k_preds)
            completeness = completeness_score(labels, k_preds)

            all_results["ari"][current_key].append(ari)
            all_results["nmi"][current_key].append(nmi)
            all_results["v_measure"][current_key].append(v_measure)
            all_results["homogeneity"][current_key].append(homogeneity)
            all_results["completeness"][current_key].append(completeness)

    return all_results


def get_unsupervised_measures(all_data, orig_data):

    ignore = ["max", "min", "avg"]

    all_results = {"dbi": {}, "chs": {},
                   "dbi_norm_sim": {}, "chs_norm": {},
                   "mul_dbi_chs": {}, "avg_dbi_chs": {}}

    for i, current_key in enumerate(all_data.keys()):

        if current_key in ignore:
            continue

        data = all_data[current_key]["m_preds"]
        truth_data = orig_data[current_key]

        all_results["dbi"][current_key] = []
        all_results["chs"][current_key] = []

        for k_preds in data:

            dbi = davies_bouldin_score(truth_data, k_preds)
            chs = calinski_harabasz_score(truth_data, k_preds)

            all_results["dbi"][current_key].append(dbi)
            all_results["chs"][current_key].append(chs)

        dbi = all_results["dbi"][current_key]
        chs = all_results["chs"][current_key]

        if i == 0:
            dbi_max, dbi_min = np.max(dbi), np.min(dbi)
            chs_max, chs_min = np.max(chs), np.min(chs)
        else:
            if np.max(dbi) > dbi_max:
                dbi_max = np.max(dbi)
            if np.min(dbi) < dbi_min:
                dbi_min = np.min(dbi)

            if np.max(chs) > chs_max:
                chs_max = np.max(chs)
            if np.min(chs) < chs_min:
                chs_min = np.min(chs)

    for i, current_key in enumerate(all_data.keys()):

        if current_key in ignore:
            continue

        all_chs = np.asarray(all_results["chs"][current_key])
        all_dbi = np.asarray(all_results["dbi"][current_key])

        all_chs = all_chs / chs_max

        all_dbi = all_dbi / dbi_max
        all_dbi = 1 - all_dbi

        all_results["mul_dbi_chs"][current_key] = all_dbi * all_chs
        all_results["avg_dbi_chs"][current_key] = (all_dbi + all_chs) / 2

        all_results["chs_norm"][current_key] = all_chs
        all_results["dbi_norm_sim"][current_key] = all_dbi

    for i, current_key in enumerate(all_results.keys()):

        if current_key in ignore:
            continue

        some_key = list(all_results[current_key].keys())[0]

        # mul = np.ones_like(all_results[current_key][some_key])
        # avg = np.zeros_like(all_results[current_key][some_key])

        maximum = np.zeros_like(all_results[current_key][some_key])

        for j, dims in enumerate(all_results[current_key].keys()):

            data = all_results[current_key][dims]

            maximum = np.maximum(maximum, data)

            # mul = mul * data
            # avg = avg + data

        # avg = avg / (j + 1)

        # all_results[current_key]["mul"] = mul
        # all_results[current_key]["avg"] = avg

        all_results[current_key]["max"] = maximum

    return all_results


def truth_validity(k_preds, labels):

    results = {"ARI": [], "NMI": [], "Homogeneity": [],
               "Completeness": [], "V-measure": []}

    for preds in k_preds:

        results["V-measure"].append(v_measure_score(labels, preds))
        results["Homogeneity"].append(homogeneity_score(labels, preds))
        results["Completeness"].append(completeness_score(labels, preds))

        results["ARI"].append(adjusted_rand_score(labels, preds))
        results["NMI"].append(normalized_mutual_info_score(labels, preds))

    return results


def no_truth_validity(k_preds, data):

    results = {"DBI": [], "CH": []}

    for preds in k_preds:
        results["DBI"].append(davies_bouldin_score(data, preds))
        results["CH"].append(calinski_harabasz_score(data, preds))

    return results


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


def hamming_distance(u, v):

    return h_distance(u, v)


def earth_movers_distance(u, v):

    return wasserstein_distance(u, v)


def create_matrix(u, choice):

    if choice == "hamming":
        measure = h_distance
    elif choice == "emd":
        measure = earth_movers_distance
    else:
        raise NotImplementedError

    num_samples, num_features = u.shape

    matrix = np.zeros((num_samples, num_samples))

    desc = "%s (F = %s)" % (choice.capitalize(), num_features)

    pbar = tqdm(total=num_samples ** 2, desc=desc)

    for i in range(num_samples):
        for j in range(num_samples):
            matrix[i, j] = measure(u[i], u[j])
            pbar.update(1)

    pbar.close()


def cosine_similarity(u, v=None):

    if v is None:
        results = cos_sim(u)
    else:
        results = cos_sim(u, v)

    return np.round(1 - results, 3)


def hellinger_distance(u):

    return 1 / np.sqrt(2) * euclidean_distance(np.sqrt(u))


def euclidean_distance(u, v=None):

    if v is None:
        results = l2_norm(u)
    else:
        results = l2_norm(u, v)

    return results


def select_measure(choice):

    if choice == 0:

        measure = euclidean_distance

    elif choice == 1:

        measure = hellinger_distance

    elif choice == 2 or choice == 3:

        measure = create_matrix

    else:

        raise NotImplementedError

    return measure
