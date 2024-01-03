"""
Author: Charlie
Purpose: General tools
"""


import os
import yaml
import shutil


def create_folder(path, overwrite=False):
    """
    Creates an overwritable user specified folder

    Parameters:
    - path (str): path to folder
    - overwrite (bool): flag for remaking existing folder
    """

    if os.path.exists(path):
        if overwrite:
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def clear_logfile(path, filename="metrics.csv"):
    """
    Clears prior log file of specific name

    Parameters:
    - path (str): path to log folder
    - filename (str): target log file
    """

    path_file = os.path.join(path, "lightning_logs", "training", filename)
    if os.path.exists(path_file):
        os.remove(path_file)


def log_params(params):
    """
    Log the parameters used for the experiment

    Parameters:
    - params (dict[str, any]): user defined parameters
    """

    print("\n---------------------------\n")

    print("Experiment Parameters")

    for current_key in params:
        print("\n%s Parameters:" % current_key.capitalize())
        for sub_key in params[current_key]:
            print("- %s: %s" % (sub_key, params[current_key][sub_key]))

    print("\n---------------------------\n")


def load_yaml(argument):
    """
    Loads YAML configuration file

    Parameters:
    - argument (str): path to configuration file

    Returns:
    - (dict[str, any]): User defined parameters
    """

    return yaml.load(open(argument), Loader=yaml.FullLoader)


def parse_args(all_args):
    """
    Parse system command line arguments

    Parameters:
    - all_args (list[str]): all system arguments caputed using "sys" library

    Returns:
    - (dict[str, str]): Formatted command line arguments
    """

    tags = ["--", "-"]

    all_args = all_args[1:]

    if len(all_args) % 2 != 0:
        print("Argument '%s' not defined" % all_args[-1])
        exit()

    results = {}

    i = 0
    while i < len(all_args) - 1:
        arg = all_args[i].lower()
        for current_tag in tags:
            if current_tag in arg:
                arg = arg.replace(current_tag, "")
        results[arg] = all_args[i + 1]
        i += 2

    return results


def load_config(sys_args):
    """
    Loads configuration file from command line argument

    Parameters:
    - sys_args (list[str]): all system arguments caputed using "sys" library

    Returns:
    - (dict[str, any]): User defined parameters
    """

    args = parse_args(sys_args)
    params = load_yaml(args["config"])
    params["cl"] = args

    return params
