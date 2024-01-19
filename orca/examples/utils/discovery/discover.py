"""
Author: Charlie
Purpose: Knowledge discovery using DL autoencoder
"""


import os

from utils.data import gather_data
from utils.general import create_folder
from utils.discovery.specific import save_neighbors
from utils.discovery.methods import comparison_matrix
from utils.discovery.extraction import get_nn_features
from utils.neural.transforms import load_data_transforms
from utils.discovery.plots import plot_features, plot_matrix
from utils.discovery.methods import find_neighbors, run_vat_and_ivat

# from utils.discovery.scoring import score_clustering
# from utils.discovery.specific import save_results, save_neighbors


def run(params):
    """
    Knowledge discovery using DL autoencoder

    Parameters:
    - params (dict[str, any]): user defined parameters
    """

    path = os.path.join(params["paths"]["discovery"], "discovery")
    create_folder(path)

    # Load: Dataset

    transforms = load_data_transforms(params["datasets"]["transforms"],
                                      params["datasets"]["interpolate"],
                                      params["datasets"]["sample_size"])

    dataset = gather_data(params["paths"]["valid"], transforms["valid"])

    # Extract: Features

    features = get_nn_features(dataset, params)

    # Plot: Features

    plot_features(path, features, dataset.labels)

    from IPython import embed
    embed()
    exit()

    # Explore: Comparison Measures

    # all_results = []

    for i in range(2):

        print("\nCurrent Measure: %s\n" % i)

        # - Gather comparison matrices

        matrix, name = comparison_matrix(features, i)
        all_matrices = run_vat_and_ivat(matrix)

        # - Find nearest neighbors

        neighbors = find_neighbors(matrix, dataset.samples)

        # - Save comparison matrices

        path_results = os.path.join(path, name)
        create_folder(path_results)

        all_tags = ["original", "vat", "ivat"]

        for tag in all_tags:
            path_save = os.path.join(path_results, tag + ".png")
            plot_matrix(path_save, all_matrices[tag], tag.capitalize())

        # - Save neighbors

        path_save = os.path.join(path_results, "neighbors")
        save_neighbors(path_save, neighbors)

        continue

        # - Gather clustering predictions

        # choices = {"measure": i, "model": 0}
        # results = run_clustering(features, choices)

        # # - Calculate clustering performance

        # scores = score_clustering(dataset.labels, results["labels"])

        # # - Save clustering partitions

        # path_save = os.path.join(path_results, results["name"])

        # save_results(path_save, dataset, results["labels"])

        # # - Update results

        # results["name"] += ", %s" % name

        # for current_key in scores:
        #     results[current_key] = scores[current_key]
        # all_results.append(results)

    # Plot: Results

    # plot_scores(path, all_results)
