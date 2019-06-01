import logging
import argparse
import yaml
import os
import subprocess
import re
import boto3
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np

from load_data import load_data

logger = logging.getLogger(__name__)


def choose_features(df1, df2, df3, df4, features_to_use=None, target=None, **kwargs):
    """Reduces each dataset to the features_to_use.
    Args:
        df1, df2, df3, df4 (:py:class:`pandas.DataFrame`): DataFrames containing the features
        features_to_use (:obj:`list`, optional): List of columnms to extract from the dataset to be features
        target (str, optional): If given, will include the target column in the output dataset as well.
    Returns:
        X1, X2, X3, X4 (:py:class:`pandas.DataFrame`): DataFrames containing extracted features (and target, it applicable)
    """

    logger.debug("Choosing features")
    if features_to_use is not None:
        # application df
        features = []
        dropped_columns = []
        for column in df1.columns:
            # Identifies if this column is in the features to use or if it is a dummy of one of the features to use
            if column in features_to_use[0] or column.split("_dummy_")[0] in features_to_use[0] or column == target:
                features.append(column)
            else:
                dropped_columns.append(column)

        if len(dropped_columns) > 0:
            logger.info("The following columns were not used as features: %s", ",".join(dropped_columns))
        logger.debug(features)
        X1 = df1[features]

        # bureau df
        features = []
        dropped_columns = []
        for column in df2.columns:
            # Identifies if this column is in the features to use or if it is a dummy of one of the features to use
            if column in features_to_use[1] or column.split("_dummy_")[0] in features_to_use[1] or column == target:
                features.append(column)
            else:
                dropped_columns.append(column)

        if len(dropped_columns) > 0:
            logger.info("The following columns were not used as features: %s", ",".join(dropped_columns))
        logger.debug(features)
        X2 = df2[features]

        # previous application df
        features = []
        dropped_columns = []
        for column in df3.columns:
            # Identifies if this column is in the features to use or if it is a dummy of one of the features to use
            if column in features_to_use[2] or column.split("_dummy_")[0] in features_to_use[2] or column == target:
                features.append(column)
            else:
                dropped_columns.append(column)

        if len(dropped_columns) > 0:
            logger.info("The following columns were not used as features: %s", ",".join(dropped_columns))
        logger.debug(features)
        X3 = df3[features]

        # installment df
        features = []
        dropped_columns = []
        for column in df4.columns:
            # Identifies if this column is in the features to use or if it is a dummy of one of the features to use
            if column in features_to_use[3] or column.split("_dummy_")[0] in features_to_use[3] or column == target:
                features.append(column)
            else:
                dropped_columns.append(column)

        if len(dropped_columns) > 0:
            logger.info("The following columns were not used as features: %s", ",".join(dropped_columns))
        logger.debug(features)
        X4 = df4[features]

    else:
        logger.debug("features_to_use is None, dfs being returned")
        X1, X2, X3, X4 = df1, df2, df3, df4

    return X1, X2, X3, X4


def choose_features_all(df, features_to_use=None, target=None, **kwargs):
    """Reduces a combined dataset to the features_to_use. Will keep the target if provided.
    Args:
        df (:py:class:`pandas.DataFrame`): DataFrame containing the features
        features_to_use (:obj:`list`, optional): List of columnms to extract from the dataset to be features
        target (str, optional): If given, will include the target column in the output dataset as well.
        **kwargs:
    Returns:
        X (:py:class:`pandas.DataFrame`): DataFrame containing extracted features (and target, it applicable)
    """

    logger.debug("Choosing features from aggregated feature file")
    if features_to_use is not None:
        features = []
        dropped_columns = []
        for column in df.columns:
            # Identifies if this column is in the features to use or if it is a dummy of one of the features to use
            if column in features_to_use or column.split("_dummy_")[0] in features_to_use or column == target:
                features.append(column)
            else:
                dropped_columns.append(column)

        if len(dropped_columns) > 0:
            logger.info("The following columns were not used as features: %s", ",".join(dropped_columns))
        logger.debug(features)
        X = df[features]
    else:
        logger.debug("features_to_use is None, df being returned")
        X = df

    return X


def get_target(df, target, save_path=None, **kwargs):
    """Get target value
    Args:
        df (:py:class:`pandas.DataFrame`): dataframe containing target
        target (str): column name of target 
        save_path (str, optional): path to save the target value
    Returns:
        y.values (numpy.ndarray): the value of target dataframe
    """

    y = df[target]

    if save_path is not None:
        y.to_csv(save_path, **kwargs)

    return y.values


def generate_features(X1, X2, X3, X4, save_features=None, **kwargs):
    """Generate features for each dataset
    Args:
        X1,X2,X3,X4 (:py:class:`pandas.DataFrame`): DataFrames containing the data to be transformed into features.
        save_features (str, optional): If given, the feature set will be saved to this path.
    Returns:
        df (py:class:`pandas.DataFrame`): A combined dataframe only containing selected features and transformed features
    """

    choose_features_kwargs = kwargs["choose_features"]
    df1, df2, df3, df4 = choose_features(X1, X2, X3, X4, **choose_features_kwargs)

    # replace DAYS_EMPLOYED: 365243 -> nan
    df1['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # generate new features for application df
    df1['DAYS_EMPLOYED_PERC'] = df1['DAYS_EMPLOYED'] / df1['DAYS_BIRTH']
    df1['PAYMENT_RATE'] = df1['AMT_ANNUITY'] / df1['AMT_CREDIT']
    df1['INCOME_CREDIT_PERC'] = df1['AMT_INCOME_TOTAL'] / df1['AMT_CREDIT']
    df1['ANNUITY_INCOME_PERC'] = df1['AMT_ANNUITY'] / df1['AMT_INCOME_TOTAL']

    # generate new features for bureau df
    aggregations = {
        'DAYS_CREDIT': ['mean'],
        'DAYS_CREDIT_ENDDATE': ['mean']
    }
    # Aggregate by application id and rename columns
    df2_agg = df2.groupby('SK_ID_CURR').agg({**aggregations})
    df2_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in df2_agg.columns.tolist()])  

    # generate new features for previous application df
    # Previous applications numeric features
    aggregations = {
        'DAYS_DECISION': ['mean'],
    }
    df3_approved = df3[df3['NAME_CONTRACT_STATUS'] == 'Approved']
    df3_agg = df3_approved.groupby('SK_ID_CURR').agg(aggregations)
    df3_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in df3_agg.columns.tolist()])

    # generate new features for installment df
    # Days days before due (no negative values)
    df4['DBD'] = df4['DAYS_INSTALMENT'] - df4['DAYS_ENTRY_PAYMENT']
    df4['DBD'] = df4['DBD'].apply(lambda x: x if x > 0 else 0)
    aggregations = {
        'DBD': ['mean'],
        'AMT_PAYMENT': ['mean'],
        'DAYS_ENTRY_PAYMENT': ['mean']
    }
    # Feature aggregation by applicant id
    df4_agg = df4.groupby('SK_ID_CURR').agg(aggregations)
    df4_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in df4_agg.columns.tolist()])

    # merge 4 dfs
    df = pd.merge(df1, df2_agg, how='left', on='SK_ID_CURR')
    df = pd.merge(df, df3_agg, how='left', on='SK_ID_CURR')
    df = pd.merge(df, df4_agg, how='left', on='SK_ID_CURR')

    if save_features is not None:
        df.to_csv(save_features, index=False)

    logger.debug("shape of features df: " + str(df.shape))

    return df


def run_features(args):
    """Orchestrates the generating of features from commandline arguments."""
    with open(args.config, "r") as f:
        config = yaml.load(f)

    if "load_data" in config:
        df1, df2, df3, df4 = load_data(config["load_data"])  
    else:
        raise ValueError("'load_data' configuration must exist in config file")

    df = generate_features(df1, df2, df3, df4, **config["generate_features"])

    if args.output is not None:
        df.to_csv(args.output)
        logger.info("Features saved to %s", args.output)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate features")
    parser.add_argument('--config', help='path to yaml file with configurations')
    parser.add_argument('--output', default=None, help="Path to CSV to save generated features")

    args = parser.parse_args()

    run_features(args)