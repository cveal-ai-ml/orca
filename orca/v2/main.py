"""
Purpose: Run ORCA Experiment
Author: Charlie
"""


import sys

import utils.discover as discover

from utils.general import load_config


def experiment(params):

    if params["experiment"] == 0:
        pass

    elif params["experiment"] == 1:
        discover.run(params)

    else:
        raise NotImplementedError


if __name__ == "__main__":

    params = load_config(sys.argv)

    experiment(params)
