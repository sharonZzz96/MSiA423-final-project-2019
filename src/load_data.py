import logging
import os
import re
import argparse
import multiprocessing
import glob
import boto3
import yaml
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import requests
import warnings
warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


def load_from_s3(sourceurl, filenames, save_path, **kwargs):
    """Download multiple CSVs from public S3, save to local, and load into multiple Pandas dataframes.
    
    Args:
        sourceurl (str): s3 bucket object url
        filenames (list): list containting the file names
        save_path (str): path to save the downloaded files
    Returns: 
        df1, df2, df3, df4 (:py:class:`pandas.DataFrame`): Four dataframes with data from the files loaded
    """

    for file in filenames:
        r = requests.get(sourceurl+file)
        open(save_path+file, 'wb').write(r.content)

    df1 = pd.read_csv(save_path+filenames[0])
    df2 = pd.read_csv(save_path+filenames[1])
    df3 = pd.read_csv(save_path+filenames[2])
    df4 = pd.read_csv(save_path+filenames[3])

    logger.info('4 files all loaded')

    return df1, df2, df3, df4


def load_data(config):
    """Load data from s3
    Args:
        config (dictionary): a configuration dictionary of load_data
    Returns:
        df1, df2, df3, df4 (:py:class:`pandas.DataFrame`): Four dataframes with data from the files loaded
    """
    how = config["how"].lower()

    # load url
    if how == "load_from_s3":
        if "load_from_s3" not in config:
            raise ValueError("'how' given as 'load_from_s3' but 'load_from_s3 not in configuration")
        else:
            df1, df2, df3, df4 = load_from_s3(**config["load_from_s3"])
            return df1, df2, df3, df4
    else:
        raise ValueError("Options for 'how' is 'load_from_s3' but %s was given" % how)


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