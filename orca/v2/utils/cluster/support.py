"""
Purpose: Clustering Support Tools
Author: Charlie
"""


import os
import pickle
import numpy as np

from tqdm import tqdm

from utils.cluster.measures import select_measure


def calculate_matrix(data, choice):

    measure = select_measure(choice)

    # Run: Euclidean or Hellinger Distance

    if choice == 0 or choice == 1:

        matrix = measure(data)

    else:

        # Set : Hamming Distance

        if choice == 2:
            name = "hamming"

        # Set: Earth Movers Distance

        else:
            name = "emd"

        # Run: Hamming or Earth Movers Distance

        matrix = measure(data, name)

    return matrix


def run_comparisons(data, choice, desc):

    results = []

    for current_key in tqdm(data.keys(), desc=desc):
        features = data[current_key]
        matrix = calculate_matrix(features, choice=choice)
        results.append(matrix)

    return np.stack(results)


def get_dist_matrices(path, all_data, create):

    path_file = os.path.join(path, "matrices.pkl")

    if create or not os.path.exists(path_file):

        all_matrices = {}

        print("\n------ Gathering Matrices ------\n")

        for current_key in all_data.keys():

            all_matrices[current_key] = {}

            # Select: Comparison Measure

            if current_key == "features":
                m_choice = 0
                desc = "Euclidean Matrices"

            elif current_key == "preds_soft":
                m_choice = 1
                desc = "Hellinger Matrices"

            elif current_key == "preds_crisp":
                m_choice = 0
                desc = "Euclidean Matrices"

            else:
                raise NotImplementedError

            # Create: Comparison Matrices
            # - Aggregate across feature sizes

            data = all_data[current_key]

            results = run_comparisons(data, m_choice, desc)

            for dims, matrix in zip(data.keys(), results):
                all_matrices[current_key][dims] = matrix

            all_matrices[current_key]["max"] = results.max(axis=0)
            all_matrices[current_key]["min"] = results.min(axis=0)
            all_matrices[current_key]["avg"] = results.mean(axis=0)

        # Save: All Matrices

        pickle.dump(all_matrices, open(path_file, "wb"))

    else:

        all_matrices = pickle.load(open(path_file, "rb"))

    return all_matrices
