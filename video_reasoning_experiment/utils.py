import logging
import os
import shutil
import yaml as yaml

from datetime import datetime


def create_experiment_dir(conf_file, exp_path, run_mode, start=None, end=None):
    """
    Creates a directory for all the outputs of the experiment (i.e. program execution).
    The directory is of shape exp_path/run_mode/config_name/timestamp_now/ (= base_path) with
    subdirectories /model and /stats.
    Also makes a snapshot of the configuration file and saves it to the created experiment directory.
    Parameters
    ----------
    conf_file : string
        Path / name of the configuration file.
    exp_path : string
        Path of the experiment directory.
    run_mode : string
        Run mode (has to be in [train, eval]).
    start : int
        Start index of the dataset. If None, the start index is not added to the experiment path name.
    end : int
        End index of the dataset. If None, the end index is not added to the experiment path name.
    Returns
    -------
    base_path : string
        The base path of the experiment directory.
    """
    config_file_name = os.path.basename(conf_file)
    dot_yaml_removal_index = 5 if "yaml" in config_file_name else 4
    config_name = config_file_name[:-dot_yaml_removal_index]
    config_name = f"{config_name}_from_{start}_to_{end}" if start is not None and end is not None else config_name
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    base_path = os.path.join(exp_path, run_mode, config_name, now)

    paths = [
        os.path.join(base_path, "states"),
    ]

    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print("Created directory", path)

    shutil.copy2(conf_file, base_path)
    print("Copied config file", conf_file, "to", base_path)

    return base_path


def parse_config(path):
    """
    Parses the configuration file located at the path that is given with the first command line argument.
    If the required command line argument is missing, the program stops.
    Parameters
    ----------
    path : string
        path to the configuration file
    Returns
    -------
    conf : dict
        The dictionary representing the configuration.
    """
    with open(path, "r") as stream:
        conf = yaml.safe_load(stream)
    return conf


def setup_logger(path, lvl=0, fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s"):
    """
    Sets up a global logger accessible via logging.getLogger("root").
    The registered logger will stream its outputs to the console as well as
    to a file out.log in the specified directory.
    Parameters
    ----------
    path : string
        Path to the folder in which to save the logfile out.log.
    lvl : int
        One of CRITICAL = 50, ERROR = 40, WARNING = 30, INFO = 20, DEBUG = 10, NOTSET = 0.
    fmt : string
        Format string representing the format of the logs.
    Returns
    -------
    root_logger : Logger
        The root logger.
    """
    log_path = os.path.join(path, "out.log")
    formatter = logging.Formatter(fmt=fmt)
    root_logger = logging.getLogger()
    root_logger.setLevel(lvl)

    root_logger.handlers = []

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return root_logger


def get_human_time(time_in_seconds: float):
    """
    Converts a time in seconds to a human readable format.
    Parameters
    ----------
    time_in_seconds : float
        Time in seconds.
    Returns
    -------
    human_time : string
        Time in human-readable format.
    """
    hours, remainder = divmod(time_in_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    human_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    return human_time
