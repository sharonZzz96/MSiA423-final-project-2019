import logging
import os
import re
import argparse
import multiprocessing
import glob
import boto3
import yaml
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


def load_csvs(file_names, directory, **kwargs):
    """Loads multiple CSVs into multiple Pandas dataframes.
    
    Args:
        file_names (list of str): List of files to load
        directory (str): Directory containing files to be loaded. 
    Returns: 
        df1, df2, df3, df4 (:py:class:`pandas.DataFrame`): Four dataframes with data from the files loaded
    """

    # Get list of files
    if file_names is None or directory is None:
        raise ValueError("filenames and directory must be given")
    elif len(file_names) != 4:
        raise ValueError("number of files to be used is wrong, need 4 files")
    else:
        subpath = os.getcwd()
        df1 = pd.read_csv(subpath+'/'+directory+file_names[0])
        df2 = pd.read_csv(subpath+'/'+directory+file_names[1])
        df3 = pd.read_csv(subpath+'/'+directory+file_names[2])
        df4 = pd.read_csv(subpath+'/'+directory+file_names[3])
        logger.info('4 files all loaded')

    return df1, df2, df3, df4


def load_data(config):
    """Load data from csvs
    Args:
        config (dictionary): a configuration dictionary of load_data
    Returns:
        df1, df2, df3, df4 (:py:class:`pandas.DataFrame`): Four dataframes with data from the files loaded
    """
    how = config["how"].lower()

    # load url
    if how == "load_csvs":
        if "load_csvs" not in config:
            raise ValueError("'how' given as 'load_csvs' but 'load_csvs' not in configuration")
        else:
            df1, df2, df3, df4 = load_csvs(**config["load_csvs"])
            return df1, df2, df3, df4
    else:
        raise ValueError("Options for 'how' is 'load_csvs' but %s was given" % how)


def run_loading(args):
    """Loads config and executes load data set
    Args:
        args: From argparse, should contain args.config
            args.config (str): Path to yaml file with load_data as a top level key containing relevant configurations
    Returns: None
    """
    with open(args.config, "r") as f:
        config = yaml.load(f)

    df1, df2, df3, df4 = load_data(**config["load_data"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--config', help='path to yaml file with configurations')

    args = parser.parse_args()

    run_loading(args)