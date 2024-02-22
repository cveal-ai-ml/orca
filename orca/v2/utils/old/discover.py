"""
Author: Charlie
Purpose: Class Discovery using Pre-trained DL Endoder
"""


from utils.models import Network
from utils.data import gather_data
from utils.network.support import get_nn_predictions


def run(params):

    path_data = params["paths"]["test"]
    path_results = params["paths"]["results"]
    load_preds = params["discovery"]["predictions"]["create"]
    percent_class_samples = params["dataset"]["percent_class_samples"]

    # Load: Dataset

    model = Network()

    # Load: Dataset

    transforms = model.weights.transforms()
    dataset = gather_data(path_data, transforms, percent_class_samples)

    # Gather: Predictions

    predictions = get_nn_predictions(path_results, model,
                                     dataset, load_preds)
