import logging
import argparse
import yaml
import os
import subprocess
import re
import datetime

import pickle

import sklearn
import xgboost
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

from load_data import load_data
from generate_features import choose_features_all, get_target

logger = logging.getLogger(__name__)


def score_model(df, path_to_tmo, threshold, save_scores=None, **kwargs):
    """Score model/predict y on test set
    Args:
        df (:py:class:`pandas.DataFrame`): a dataframe containing all selected features used in training
        path_to_tmo (str): path to the saved model
        threshold (float): threshold used in classificaiton model
        save_scores (str, optional): path to save the predicted score
    Returns:
        result (:py:class:`pandas.DataFrame`): a dataframe containing predicted class and probability
    """

    # load the model
    with open(path_to_tmo, "rb") as f:
        model = pickle.load(f)

    # only keep the chosen features in dataframe if specified
    if "choose_features_all" in kwargs:
        X = choose_features_all(df, **kwargs["choose_features_all"])
    else:
        X = df

    # predict probability and class label based on threshold
    ypred_proba_test = model.predict_proba(X)[:,1]
    result = pd.DataFrame(ypred_proba_test)
    result = result.rename(columns={0: "ypred_proba_test"})
    result['ypred_bin_test'] = np.nan
    result['ypred_bin_test'][result['ypred_proba_test'] > threshold] = 1
    result['ypred_bin_test'][result['ypred_proba_test'] <= threshold] = 0

    # save predicted score if specified
    if save_scores is not None:
        pd.DataFrame(result.to_csv(save_scores, index=False))
        logger.info('prediction result saved')

    return result


def run_scoring(args):
    with open(args.config, "r") as f:
        config = yaml.load(f)

    if args.input is not None:
        df = pd.read_csv(args.input)
    elif "train_model" in config and "split_data" in config["train_model"] and "save_split_prefix" in config["train_model"]["split_data"]:
        try:
            df = pd.read_csv(config["train_model"]["split_data"]["save_split_prefix"] + "-test-features.csv")
        except:
            raise FileNotFoundError("run train_model.py first to split data")
    else:
        raise ValueError("Path to CSV for input data must be provided through --input or "
                         "'train_model' configuration must exist in config file")

    result = score_model(df, **config["score_model"])

    if args.output is not None:
        pd.DataFrame(result).to_csv(args.output, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Score model")
    parser.add_argument('--config', '-c', help='path to yaml file with configurations')
    parser.add_argument('--input', '-i', default=None, help="Path to CSV for input to model scoring")
    parser.add_argument('--output', '-o', default=None, help='Path to where the scores should be saved to (optional)')

    args = parser.parse_args()

    run_scoring(args)
