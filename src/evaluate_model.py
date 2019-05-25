import logging
import argparse
import yaml
import os
import subprocess
import re
import datetime

import sklearn
import pandas as pd
import numpy as np

from sklearn import metrics

logger = logging.getLogger(__name__)


def evaluate_model(df, y_predicted, **kwargs):
    """Evaluate the performance of the model   
    Args:
        df (:py:class:`pandas.DataFrame`): a dataframe containing true y label
        y_predicted (:py:class:`pandas.DataFrame`): a dataframe containing predicted probability and score
    Returns: 
        confusion_df (:py:class:`pandas.DataFrame`): a dataframe reporting confusion matrix
    """

    ypred_proba_test = y_predicted['ypred_proba_test']
    ypred_bin_test = y_predicted['ypred_bin_test']

    y_true = df.iloc[:,0]
    # calculate auc and accuracy if specified
    if "auc" in kwargs["metrics"]:
        auc = sklearn.metrics.roc_auc_score(df, ypred_proba_test)
        print('AUC on test: %0.3f' % auc)
    if "accuracy" in kwargs["metrics"]:
        accuracy = sklearn.metrics.accuracy_score(df, ypred_bin_test)
        print('Accuracy on test: %0.3f' % accuracy)

    # generate confusion matrix and classification report
    confusion = sklearn.metrics.confusion_matrix(y_true, ypred_bin_test)
    classification_report = sklearn.metrics.classification_report(y_true, ypred_bin_test)
    confusion_df = pd.DataFrame(confusion,
        index=['Actual negative','Actual positive'],
        columns=['Predicted negative', 'Predicted positive'])
    
    print(confusion_df)
    print(classification_report)

    return confusion_df


def run_evaluation(args):
    with open(args.config, "r") as f:
        config = yaml.load(f)

    if args.input is not None:
        df = pd.read_csv(args.input)
    elif "train_model" in config and "split_data" in config["train_model"] and "save_split_prefix" in config["train_model"]["split_data"]:
        df = pd.read_csv(config["train_model"]["split_data"]["save_split_prefix"]+ "-test-targets.csv")
        print('true' + str(df.shape))
        logger.info("test target loaded")
    else:
        raise ValueError("Path to CSV for input data must be provided through --input or "
                         "'train_model' configuration must exist in config file")

    if "score_model" in config and "save_scores" in config["score_model"]:
        y_predicted = pd.read_csv(config["score_model"]["save_scores"])
        print('predict' + str(y_predicted.shape))
        logger.info("test predict loaded")
    else:
        raise ValueError("'score_model' configuration mush exist in config file")

    confusion_df = evaluate_model(df, y_predicted, **config["evaluate_model"])
    if args.output is not None:
        confusion_df.to_csv(args.output, index=False)
    elif "evaluate_model" in config and "save_evaluation" in config["evaluate_model"]:
        confusion_df.to_csv(config["evaluate_model"]["save_evaluation"], index=False)
    else:
        raise ValueError("Path to CSV for ouput data must be provided through --output or "
                         "'evaluate_model' configuration must exist in config file")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument('--config', help='path to yaml file with configurations')
    parser.add_argument('--input', default=None, help="Path to CSV for input to model scoring")
    parser.add_argument('--output', default=None, help="Path to CSV for output to confusion matrix")

    args = parser.parse_args()

    run_evaluation(args)
