"""
Purpose: Class Discovery using Pre-trained DL Network
Author: Charlie
"""


from utils.data import gather_data
from utils.network.models import Network
from utils.cluster.methods import run_clustering
from utils.cluster.support import get_dist_matrices
from utils.network.support import get_nn_predictions
from utils.plots import plot_matrices, plot_clustering
from utils.plots import plot_features, plot_confidences


def run(params):

    path_results = params["paths"]["results"]

    # Load: Network

    device = params["system"]["gpus"]["accelerator"]
    batch_size = params["discovery"]["predictions"]["batch_size"]

    model = Network(device, batch_size)

    # Load: Dataset

    path_data = params["paths"]["test"]
    percent = params["discovery"]["dataset"]["percent_class_samples"]

    transforms = model.weights.transforms()
    dataset = gather_data(path_data, transforms, percent)

    # Gather: Predictions

    all_top_k = params["discovery"]["predictions"]["top_k"]
    create_preds = params["discovery"]["predictions"]["create"]
    tsne_dims = params["discovery"]["predictions"]["tsne_dims"]

    preds = get_nn_predictions(path_results, model, dataset,
                               all_top_k, tsne_dims, create_preds)

    if create_preds:

        plot_features(path_results, preds["features"], dataset.labels)

        """
        plot_confidences(path_results,
                         preds["preds_soft"],
                         dataset.labels, "preds_soft")

        plot_confidences(path_results,
                         preds["preds_crisp"],
                         dataset.labels, "preds_crisp")
        """

    # Gather: Distance Matrices

    create_matrices = params["discovery"]["matrices"]["create"]

    matrices = get_dist_matrices(path_results, preds, create_matrices)

    if create_matrices:
        plot_matrices(path_results, matrices)

    # Run: Clustering

    create_clusters = params["discovery"]["clustering"]["create"]
    max_k = params["discovery"]["clustering"]["max_k"]

    results = run_clustering(path_results, matrices, max_k,
                             dataset.labels, create_clusters)

    plot_clustering(path_results, results, preds, dataset)

    from IPython import embed
    embed()
    exit()
