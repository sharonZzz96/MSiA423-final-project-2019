import argparse
import logging
import pickle

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
import yaml
import xgboost

from generate_features import choose_features_all, get_target

logger = logging.getLogger(__name__)

methods = dict(xgboost=xgboost.XGBClassifier)


def split_data(X, y, train_size=1, test_size=0, random_state=123, save_split_prefix=None):
    """
    Args:
        X (:py:class:`pandas.DataFrame` or :py:class:`numpy.Array`): Features to be split
        y (:py:class:`pandas.Series` or :py:class:`numpy.Array`): Target to be split
        train_size (`float`): Fraction of dataset to use for training. Default 1 (all data). Must be between 0 and 1.
        test_size (`float`): Fraction of dataset to use for testing. Default 0 (no data). Must be between 0 and 1.
        random_state (`int`): Integer value to choose random seed.
        save_split_prefix (str, optional): If given, the datasets will be saved with the given prefix, which can include
            the path to the directory for saving plus a prefix for the file, e.g. `data/features/2019-05-01-` will
            result in files saved to `data/features/2019-05-01-train-features.csv`,
            `data/features/2019-05-01-train-targets.csv`, and similar files for `test` and `validate` if `test_size`
            and/or `validate_size` are greater than 0.
    Returns:
        X (dict): Dictionary where keys are train, test and values are the X features for those splits.
        y (dict): Dictionary where keys are train, test and values are the y targets for those splits.
    """
    if y is not None:
        assert len(X) == len(y)
        include_y = True
    else:
        y = [0] * len(X)
        include_y = False
    if train_size + test_size == 1:
        prop = True
    elif train_size + test_size == len(X):
        prop = False
    else:
        raise ValueError("train_size + test_size  "
                         "must equal 1 or equal the number of rows in the dataset")

    if prop:
        train_size = int(np.round(train_size * len(X)))
        test_size = int(len(X) - train_size)

    if train_size == 1:
        X_train, y_train = X, y
        X_test, y_test = [], []
    elif test_size == 0:
        X_train, y_train = X, y
        X_test, y_test = [], []
    else:
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y,
                                                                                        test_size=test_size,
                                                                                        random_state=random_state)
    X = dict(train=X_train)
    y = dict(train=y_train)

    if len(X_test) > 0:
        X["test"] = X_test
        y["test"] = y_test

    if save_split_prefix is not None:
        for split in X:
            pd.DataFrame(X[split]).to_csv("%s-%s-features.csv" % (save_split_prefix, split), index=False)
            if include_y:
                pd.DataFrame(y[split]).to_csv("%s-%s-targets.csv" % (save_split_prefix, split), index=False)

            logger.info("X_%s and y_%s saved to %s-%s-features.csv and %s-%s-targets.csv",
                        split, split,
                        save_split_prefix, split,
                        save_split_prefix, split)

    if not include_y:
        y = dict(train=None)

    return X, y


def train_model(df, method=None, save_tmo=None, **kwargs):
    """Train a xgboost classification model
    Args:
        df (:py:class:`pandas.DataFrame`): a dataframe containing all data
        method (str, optional): model name
        save_tmo (str, optional): path to save the model
    Returns:
        model (obj): a xgboost classification model
    """

    assert method in methods.keys()  # `methods` defined at top of file, possible methods for training

    # If "get_target" in the config file under "train_model", will get the target data for supervised learning
    # Otherwise y = None and the model must be unsupervised.
    if "get_target" in kwargs:
        y = get_target(df, **kwargs["get_target"])
        df = df.drop(labels=[kwargs["get_target"]["target"]], axis=1)
    else:
        y = None

    # If "choose_features" in the config file under "train_model", will reduce the feature set to those listed
    if "choose_features_all" in kwargs:
        X = choose_features_all(df, **kwargs["choose_features_all"])
    else:
        X = df

    # Splits the training data according to the "split_data" parameters. If this is an empty dictionary
    # (from prior step, because it is not in the configuration file), then the full dataset is returned (train_size=1)
    X, y = split_data(X, y, **kwargs["split_data"])

    if "parameter" in kwargs:
        # Instantiates a model class for the training `method` provided
        model = methods[method](objective =kwargs['parameter']['objective'], n_estimators=kwargs['parameter']['n_estimators'], 
            learning_rate =kwargs['parameter']['learning_rate'], max_depth =kwargs['parameter']['max_depth'])
    else:
        raise ValueError('paramter must be specified in config')

    # Fit the model with the training data
    model.fit(X["train"], y["train"], eval_set=[(X["test"], y["test"])],
        eval_metric='auc', verbose=50, early_stopping_rounds=20)

    # Save the trained model object
    if save_tmo is not None:
        with open(save_tmo, "wb") as f:
            pickle.dump(model, f)
        logger.info("Trained model object saved to %s", save_tmo)

    return model


def run_training(args):
    """Orchestrates the training of the model using command line arguments."""

    with open(args.config, "r") as f:
        config = yaml.load(f)

    logger.info("Training configuration file, %s, loaded", args.config)

    if args.input is not None:
        df = pd.read_csv(args.input)
        logger.info("Features for input into model loaded from %s", args.input)
    elif "generate_features" in config and "save_features" in config["generate_features"]:
        df = pd.read_csv(config["generate_features"]["save_features"])
        logger.info("Features for input into model loaded from %s", config["generate_features"]["save_features"])
    else:
        raise ValueError("Path to CSV for input data must be provided through --input or "
                         "'load_data' configuration must exist in config file")

    tmo = train_model(df, **config["train_model"])

    # save the model
    if args.output is not None:
        with open(args.output, "wb") as f:
            pickle.dump(tmo, f)
        logger.info("Trained model object saved to %s", args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument('--config', default="config/test_model_config.yml",
                        help='path to yaml file with configurations')
    parser.add_argument('--input', default=None, help="Path to CSV for input to model training")
    parser.add_argument('--output', default=None, help='Path to where the dataset should be saved to (optional')

    args = parser.parse_args()

    run_training(args)

