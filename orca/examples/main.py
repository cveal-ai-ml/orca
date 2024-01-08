"""
Author: Charlie
Purpose: Run experiment using DL autoencoder
"""


import sys

import utils.neural.trainer as trainer
import utils.discovery.discover as discover

from utils.general import load_config, configure_params


def run_experiment(params):
    """
    Run experiment using DL autoencoder

    Parameters:
    - params (dict[str, any]): user defined parameters
    """

    # Train: DL Autoencoder

    if params["goal"]["experiment"] == 0:
        trainer.run(params)

    # Perform: Autoencoder Driven Knowledge Discovery

    elif params["goal"]["experiment"] == 1:
        discover.run(params)

    else:
        raise NotImplementedError


if __name__ == "__main__":

    """
    Setup experiment
    """

    # Load: User Defined Parameters

    params = load_config(sys.argv)

    # Update: YAML Parameters

    configure_params(params)

    # Run: Experiment

    run_experiment(params)
