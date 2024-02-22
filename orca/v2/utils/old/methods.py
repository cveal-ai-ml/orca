"""
Author: Charlie
Purpose: Useful methods
"""

import os
import shutil
import pickle
import numpy as np

from tqdm import tqdm
from sklearn.metrics import rand_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

from sklearn_extra.cluster import KMedoids
from sklearn.cluster import MiniBatchKMeans

from utils.general import create_folder
from utils.extraction import get_embeddings
from utils.measures import compare_neighbors
from utils.measures import kl_divergence, cosine_similarity
from utils.measures import euclidean_distance, hellinger_distance
from utils.plots import plot_matrix, plot_features, plot_k_clustering


def save_clusters(path, all_samples, all_labels):

    for current_label in np.unique(all_labels):
        indices = np.where(current_label == all_labels)
        class_samples = all_samples[indices]

        path_folder = os.path.join(path, str(current_label).zfill(3))
        create_folder(path_folder, overwrite=True)

        for i, sample in enumerate(class_samples):
            filename = sample.split("/")[-1]
            path_file = os.path.join(path_folder, filename)
            shutil.copyfile(sample, path_file)


def get_preds_from_exemplars(data, all_exemplars, choice,
                             t_max=0.95, t_min=0.50, min_size=10):

    # Choice: K-Means

    if choice == 0:
        comparison = euclidean_distance

    # Choice: K-Medoids
    elif choice == 1:
        comparison = cosine_similarity

    # Choice: Undefined

    else:
        raise NotImplementedError

    memberships = np.zeros((data.shape[0], len(all_exemplars)))

    for i, exemplar in enumerate(all_exemplars):
        exemplar = exemplar.reshape(1, -1)
        d_sims = comparison(data, exemplar)

        d_sims = (d_sims - d_sims.min()) / (d_sims.max() - d_sims.min())
        sims = 1 - d_sims

        memberships[:, i] = sims.reshape(-1)

    t = t_max
    keep_going = 1
    while keep_going:
        predictions = []
        for confidences in memberships:
            if (confidences < t).all():
                predictions.append(-1)
            else:
                label = np.argmax(confidences)
                predictions.append(label)

        counts = []
        for label in np.unique(predictions):
            if label == -1:
                continue
            counts.append(len(np.where(label == predictions)[0]))
        counts = np.asarray(counts)

        if (counts > min_size).all() or t < t_min:
            keep_going = 0
            if t < t_min:
                predictions = []
        else:
            t = t - 0.05

    return predictions, t


def run_k_clustering(data, labels, alg_choice, k=20):

    all_results = {"cluster_preds": [], "exemplar_preds": [],
                   "dbi": [], "ch": [], "rand": [], "thresholds": []}

    if alg_choice == 0:
        name = "K-Means"
    elif alg_choice == 1:
        name = "K-Mediods"
    else:
        raise NotImplementedError

    pbar = tqdm(total=k-3, desc="Using %s Algorithm" % name)

    for i in range(3, k, 1):

        if alg_choice == 0:
            model = MiniBatchKMeans(n_clusters=i, n_init="auto").fit(data)
            # c_preds = model.fit_predict(data)
            exemplars = model.cluster_centers_

        elif alg_choice == 1:
            model = KMedoids(n_clusters=i, metric="cosine").fit(data)
            # c_preds = model.fit_predict(data)
            exemplars = np.asarray([data[i] for i in model.medoid_indices_])
        else:
            raise NotImplementedError

        e_preds, t = get_preds_from_exemplars(data, exemplars, alg_choice)

        if len(e_preds) == 0:
            pbar.update(k-i)
            break

        subset_data = np.asarray([data[i] for i in range(data.shape[0])
                                  if e_preds[i] != -1])

        subset_preds = [ele for ele in e_preds if ele != -1]
        subset_labels = np.asarray([labels[i] for i in range(data.shape[0])
                                   if e_preds[i] != -1])

        ch = calinski_harabasz_score(subset_data, subset_preds)
        dbi = davies_bouldin_score(subset_data, subset_preds)
        rand = rand_score(subset_labels, subset_preds)

        all_results["ch"].append(ch)
        all_results["dbi"].append(dbi)
        all_results["rand"].append(rand)
        all_results["thresholds"].append(t)
        # all_results["cluster_preds"].append(c_preds)
        all_results["exemplar_preds"].append(e_preds)

        pbar.update(1)

    dbi = np.asarray(all_results["dbi"])
    rand = np.asarray(all_results["rand"])

    if len(dbi) > 0:
        dbi_norm = dbi / dbi.max()
        all_results["dbi_norm"] = dbi_norm

    if len(dbi) > 0 and len(rand) > 0:

        fuse_dbi_rand_mul = (1 - dbi_norm) * rand
        fuse_dbi_rand_avg = ((1 - dbi_norm) + rand) / 2

        all_results["fuse_dbi_rand_mul"] = fuse_dbi_rand_mul
        all_results["fuse_dbi_rand_avg"] = fuse_dbi_rand_avg

    return all_results


def get_clusters(path, all_results, dataset, max_clusters=20):

    tag_a = "2048 Features (NN)"
    tag_b = "2 Features (TSNE)"
    tag_c = "Confidence Distribution"

    all_data = {}
    all_data[tag_a] = all_results["original"]["features"][tag_a]
    all_data[tag_b] = all_results["original"]["features"][tag_b]
    all_data[tag_c] = all_results["original"]["distributions"][tag_c]

    path_root = os.path.join(path, "clustering")

    # pbar = tqdm(total=len(all_data.keys()), desc="Clustering Datasets")

    for i, current_key in enumerate(all_data):

        print("\nCurrent Dataset: %s\n" % current_key)

        path_folder = os.path.join(path_root, current_key)

        data = all_data[current_key]

        # Cluster via K Algorithms

        all_tags = ["k_means", "k-medoids"]

        for j, tag in enumerate(all_tags):
            path_save = os.path.join(path_folder, tag)
            create_folder(path_save, overwrite=True)

            k_results = run_k_clustering(data, dataset.labels,
                                         alg_choice=j, k=max_clusters)

            title = "%s Clustering Analytics" % tag.capitalize()
            plot_k_clustering(path_save, k_results, title)

            for k in range(len(k_results["exemplar_preds"])):
                path_k = os.path.join(path_save, "exemplar_preds", str(k+2).zfill(3))
                create_folder(path_k, overwrite=True)
                preds_k = k_results["exemplar_preds"][k]
                save_clusters(path_k, dataset.samples, preds_k)


def make_crisp(predictions):

    return np.asarray([np.argmax(ele) for ele in predictions])


def get_nn_predictions(path, model, dataset, create):

    path_file = os.path.join(path, "preds.pkl")

    if create or not os.path.exists(path_file):
        create_folder(path, overwrite=True)
        predictions = model.test(dataset)
        pickle.dump(predictions, open(path_file, "wb"))
    else:
        predictions = pickle.load(open(path_file, "rb"))

    return predictions


def save_comparisons(path, data, thresholds):

    for i, c_name in enumerate(data.keys()):
        all_matrices = data[c_name]
        for j, m_name in enumerate(all_matrices.keys()):
            matrix = all_matrices[m_name]

            title = "%s - %s" % (c_name, m_name)
            path_file = os.path.join(path, "original", "%s.png" % title)

            plot_matrix(path_file, matrix, "%s" % title)

            t = thresholds[i][j]

            matrix = 1 - matrix
            indices_pos = np.where(matrix >= t)
            indices_neg = np.where(matrix < t)

            matrix[indices_pos] = 1
            matrix[indices_neg] = 0

            title += " (Thresh=%s)" % t
            path_file = os.path.join(path, "threshold", "%s.png" % title)
            plot_matrix(path_file, matrix, "%s" % title)


def get_comparison_matrix(data, all_measures, iterative=False):

    num_samples = data.shape[0]

    all_matrices = {}
    for m_name in all_measures:

        measure = all_measures[m_name]

        if iterative:

            desc = "Creating %s Matrix" % m_name
            pbar = tqdm(total=num_samples ** 2, desc=desc)
            matrix = np.zeros((num_samples, num_samples))

            for i in range(num_samples):
                for j in range(num_samples):
                    matrix[i][j] = measure(data[i], data[j])
                    pbar.update(1)

            pbar.close()

        else:

            if m_name == "KLD":
                data = data + 1e-7
                desc = "Creating %s Matrix" % m_name
                pbar = tqdm(total=num_samples, desc=desc)
                matrix = np.zeros((num_samples, num_samples))
                for i in range(num_samples):
                    matrix[i, :] = measure(data[i], data)
                    pbar.update(1)
                pbar.close()
            else:
                matrix = measure(data)

        all_matrices[m_name] = matrix / np.max(matrix)

    return all_matrices


def run_comparisons(path, all_features, all_measures, iterative=False):

    all_results = {}
    for f_name in all_features.keys():
        features = all_features[f_name]
        all_results[f_name] = get_comparison_matrix(features, all_measures,
                                                    iterative=iterative)
    return all_results


def create_matrix_folders(path):

    for tag in ["original", "threshold"]:
        path_folder = os.path.join(path, tag)
        create_folder(path_folder, overwrite=True)


def get_comparisons(path, predictions, truth_labels, create, n_size=5):

    path_final = os.path.join(path, "comparisons.pkl")

    if not create and os.path.exists(path_final):

        all_results = pickle.load(open(path_final, "rb"))

    else:

        confidences = predictions["confidences"]
        features = predictions["features"]

        features = (features - features.min()) / (features.max() - features.min())

        # Compare: Spatial Feature

        path_folder = os.path.join(path, "explore", "features")
        create_matrix_folders(path_folder)

        tsne, pca = get_embeddings(features)
        pca = (pca - pca.min()) / (pca.max() - pca.min())
        tsne = (tsne - tsne.min()) / (tsne.max() - tsne.min())

        # - Rows correspond to features, columns correspond to measures

        all_thresholds = [[0.45, 0.60],
                          [0.95, 0.98],
                          [0.95, 0.98]]

        all_features = {"2048 Features (NN)": features,
                        "2 Features (TSNE)": tsne, "2 Features (PCA)": pca}

        all_measures = {"Euclidean": euclidean_distance,
                        "Cosine": cosine_similarity}

        for f_name in all_features:
            path_file = os.path.join(path_folder, "%s Scatter.png" % f_name)
            plot_features(path_file, all_features[f_name], truth_labels, f_name)

        feature_results = run_comparisons(path_folder,
                                          all_features,
                                          all_measures)

        save_comparisons(path_folder, feature_results, all_thresholds)

        # Compare: Prediction Neighborhood

        path_folder = os.path.join(path, "explore", "neighborhood")
        create_matrix_folders(path_folder)

        preds = []
        for data in confidences:
            indices = np.argsort(data)[::-1][:n_size]
            preds.append(indices)
        preds = np.asarray(preds)

        all_thresholds = [[0.50]]
        all_neighbors = {"Pred Labels, N=%s" % n_size: preds}
        all_measures = {"Neighborhood Alg": compare_neighbors}

        neighbor_results = run_comparisons(path_folder, all_neighbors,
                                           all_measures, iterative=True)

        save_comparisons(path_folder, neighbor_results, all_thresholds)

        # Compare: Prediction Distributions

        path_folder = os.path.join(path, "explore", "distribution")
        create_matrix_folders(path_folder)

        all_thresholds = [[0.4, 0.5, 0.9]]

        all_distributions = {"Confidence Distribution": confidences}
        all_measures = {"Cosine": cosine_similarity,
                        "Hellinger": hellinger_distance,
                        "KLD": kl_divergence}

        distri_results = run_comparisons(path_folder,
                                         all_distributions,
                                         all_measures)

        save_comparisons(path_folder, distri_results, all_thresholds)

        # Combine: Comparison Matrices

        path_folder = os.path.join(path, "explore", "combined")
        create_matrix_folders(path_folder)

        a = feature_results["2048 Features (NN)"]["Cosine"]
        b = feature_results["2 Features (TSNE)"]["Euclidean"]
        c = neighbor_results["Pred Labels, N=%s" % n_size]["Neighborhood Alg"]
        d = distri_results["Confidence Distribution"]["KLD"]

        combined_add = a + b + c + d
        combined_mul = a * b * c * d
        combined_avg = combined_add / 4

        combined_add = combined_add / np.max(combined_add)
        combined_mul = combined_mul / np.max(combined_mul)
        combined_avg = combined_avg / np.max(combined_avg)

        combined_max = np.zeros_like(a)
        combined_min = np.ones_like(a)
        for matrix in [a, b, c, d]:
            combined_max = np.maximum(combined_max, matrix)
            combined_min = np.minimum(combined_min, matrix)

        all_matrices = {"Addition": combined_add,
                        "Average": combined_avg,
                        "Maximum": combined_max,
                        "Minimum": combined_min,
                        "Multiply": combined_mul}

        combined_results = {"Combined Results": all_matrices}

        all_thresholds = [[0.7, 0.75, 0.5, 0.95, 0.99]]

        save_comparisons(path_folder, combined_results, all_thresholds)

        all_features = {"features": all_features,
                        "neighbors": all_neighbors,
                        "distributions": all_distributions}

        all_matrices = {"features": feature_results,
                        "neighbors": neighbor_results,
                        "distributions": distri_results,
                        "combined": combined_results}

        all_results = {"original": all_features, "matrices": all_matrices}

        pickle.dump(all_results, open(path_final, "wb"))

    return all_results
