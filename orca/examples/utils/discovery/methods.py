"""
Author: Charlie
Purpose: Image comparison methods
"""

import numpy as np

from tqdm import tqdm

from utils.discovery.measures import select_measure


def comparison_matrix(dataset, choice):

    function, name = select_measure(choice)

    pbar = tqdm(total=len(dataset) ** 2, desc="%s Matrix" % name.capitalize())

    matrix = np.zeros((len(dataset), len(dataset)))
    for i in range(len(dataset)):
        u = dataset[i]
        for j in range(len(dataset)):
            v = dataset[j]
            matrix[i, j] = function(u, v)

            pbar.update(1)

    pbar.close()

    matrix = matrix / matrix.max()

    return matrix, name


def find_neighbors(matrix, all_samples, k=5):

    all_results = []
    for i, row in enumerate(matrix):
        indices = np.argsort(row)[:k]

        from IPython import embed
        embed()
        exit()

        group = all_samples[indices]
        results = {"target": all_samples[i], "group": group}
        all_results.append(results)

    return all_results


def ivat(matrix, verbose=0):
    """
    Calculates improved Visual Assessment Tendency (iVAT)

    Parameters:
    - matrix (np.ndarray[float]): VAT matrix
    - verbose (int): flag to show progress bar

    Returns:
    - (np.ndarray[float]): iVAT matrix
    """

    ivat_matrix = np.zeros_like(matrix)

    if verbose:
        pbar = tqdm(total=matrix.shape[0], desc="Calculating iVAT")

    for r in range(1, matrix.shape[0]):

        j = np.argmin(matrix[r, :r])
        ivat_matrix[r, j] = matrix[r, j]

        for c in range(r):
            ivat_matrix[r, c] = max(matrix[r, j], ivat_matrix[j, c])

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ivat_matrix[i, j] = ivat_matrix[j, i]

        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()

    return ivat_matrix


def vat(matrix, verbose=0):
    """
    Calculates Visual Assessment Tendency (VAT)

    Parameters:
    - matrix (np.ndarray[float]): dissimilarity measure matrix
    - verbose (int): flag to show progress bar

    Returns:
    - (tuple[np.ndarray[float], list[int]): VAT matrix & sorted indices
    """

    assert len(matrix.shape) == 2, "VAT must be a one dimensional matrix"
    assert matrix.shape[0] == matrix.shape[1], "VAT shape must be squared"

    # Select: Origin Node

    i = np.argmax(matrix)

    i, j = np.unravel_index(i, matrix.shape)

    # Define: Sets I, J

    all_i = []
    all_j = list(range(matrix.shape[0]))

    # Update: Sets I, J

    all_i.append(i)
    all_j.remove(i)

    # Calculate: Ordered Dissimilarity Image (ODI)

    p = np.zeros(matrix.shape[0]).reshape(-1).astype(int)
    p[0] = i

    if verbose:
        pbar = tqdm(total=matrix.shape[0], desc="Calculating VAT")

    for z in range(1, matrix.shape[0]):

        all_values, indices = [], []
        for i in all_i:
            for j in all_j:
                indices.append([i, j])
                all_values.append(matrix[i, j])

        all_values = np.asarray(all_values)

        index = np.argmin(all_values)

        i, j = indices[index]

        p[z] = j
        all_i = all_i + [j]
        all_j.remove(j)

        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()

    # Finalize: ODI

    vat_matrix = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            vat_matrix[i, j] = matrix[p[i], p[j]]

    return (vat_matrix, all_i)


def run_vat_and_ivat(data):
    """
    Generate VAT and iVAT matrices from dissimilarity matrix

    Parameters:
    - data (np.ndarray[float]): dissimilarity measure matrix

    Returns:
    - (dict[str, any]): VAT, iVAT, and meta information
    """

    vat_matrix, indices = vat(data)
    ivat_matrix = ivat(vat_matrix)

    results = {}
    results["original"] = data
    results["vat"] = vat_matrix
    results["indices"] = indices
    results["ivat"] = ivat_matrix

    return results
