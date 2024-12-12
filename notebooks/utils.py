"""
utils.py

This module contains utility functions for the ML_ClimateEnergy_project2024.

Author: Tristan Waddington
Date: nov 2024

"""

###############################################################################
# Imports
###############################################################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler


###############################################################################
# Specific Train/Test/Val split
###############################################################################
def split_train_test_val(df, end_train_date: str, end_test_date: str):
    """Split the DataFrame into train, test and validation sets.
    df: DataFrame to split.
    end_train_date: (str) last date of the train set in format 'YYYY-MM-DD' excluded.
    end_test_date: (str) last date of the test set in format 'YYYY-MM-DD' excluded.
    returns: df_train, df_test, df_val.
    """
    end_test_date = pd.to_datetime(end_test_date)
    end_train_date = pd.to_datetime(end_train_date)
    assert (
        end_train_date < end_test_date
    ), "The test date should be after the train date"
    assert (
        end_test_date < df.index.max()
    ), "The test date should be before the last date"
    df_train = df.loc[df.index < end_train_date]
    df_test = df.loc[(df.index >= end_train_date) & (df.index < end_test_date)]
    df_val = df.loc[df.index >= end_test_date]
    return df_train, df_test, df_val


###############################################################################
# Loading and preparing data
###############################################################################


def load_normalize_data():
    """Load the previous prepared normalized dataset
    returns: X_train, X_test, y_train, Y_test"""
    # Load variable_datasets_X_train from HDF5
    variable_datasets_X_train = {}
    with pd.HDFStore("data/norm_data/variable_datasets_X_train.h5") as store:
        for variable in store.keys():
            variable_name = variable.strip("/")
            variable_datasets_X_train[variable_name] = store[variable]

    # Load variable_datasets_X_test from HDF5
    variable_datasets_X_test = {}
    with pd.HDFStore("data/norm_data/variable_datasets_X_test.h5") as store:
        for variable in store.keys():
            variable_name = variable.strip("/")
            variable_datasets_X_test[variable_name] = store[variable]

    # Load variable_datasets_y_train from HDF5
    variable_datasets_y_train = {}
    with pd.HDFStore("data/norm_data/variable_datasets_y_train.h5") as store:
        for variable in store.keys():
            variable_name = variable.strip("/")
            variable_datasets_y_train[variable_name] = store[variable]

    # Load variable_datasets_y_test from HDF5
    variable_datasets_y_test = {}
    with pd.HDFStore("data/norm_data/variable_datasets_y_test.h5") as store:
        for variable in store.keys():
            variable_name = variable.strip("/")
            variable_datasets_y_test[variable_name] = store[variable]

    return (
        variable_datasets_X_train,
        variable_datasets_X_test,
        variable_datasets_y_train,
        variable_datasets_y_test,
    )


def load_red_data():
    """Load the previous prepared reduced dataset and split it into train, test and validation sets.
    returns: X_train, X_test, X_val, y_train, y_test, y_val"""
    # Read the data
    X_red = pd.read_csv("data/red_data/X_red.csv", index_col=0, parse_dates=True)
    Y_red = pd.read_csv("data/red_data/Y_red.csv", index_col=0, parse_dates=True)
    # Split the data into train, test and validation sets
    end_train_date = "2016-01-01"
    end_test_date = "2019-01-01"
    X_train, X_test, X_val = split_train_test_val(X_red, end_train_date, end_test_date)
    Y_train, Y_test, Y_val = split_train_test_val(Y_red, end_train_date, end_test_date)

    return X_train, X_test, X_val, Y_train, Y_test, Y_val


def load_red_norm_data():
    """Load the previous prepared reduced dataset and split it into train, test and validation sets.
    Then normalize the data.
    returns: X_train, X_test, X_val, y_train, y_test, y_val, scalers"""
    # Read the data
    X_red = pd.read_csv("data/red_data/X_red.csv", index_col=0, parse_dates=True)
    Y_red = pd.read_csv("data/red_data/Y_red.csv", index_col=0, parse_dates=True)
    # Split the data into train, test and validation sets
    end_train_date = "2016-01-01"
    end_test_date = "2019-01-01"
    X_train, X_test, X_val = split_train_test_val(X_red, end_train_date, end_test_date)
    Y_train, Y_test, Y_val = split_train_test_val(Y_red, end_train_date, end_test_date)
    X_train_norm, X_test_norm, X_val_norm, Y_train_norm, Y_test_norm, Y_val_norm = (
        X_train.copy(),
        X_test.copy(),
        X_val.copy(),
        Y_train.copy(),
        Y_test.copy(),
        Y_val.copy(),
    )
    # Normalize the data by identic features, use robust scalers to avoid outliers
    scalers = {
        "blh_scaler": RobustScaler(),
        "d2m_scaler": RobustScaler(),
        "skt_saler": RobustScaler(),
        "sp_scaler": RobustScaler(),
        "ssrd_scaler": RobustScaler(),
        "t2m_scaler": RobustScaler(),
        "tcc_scaler": RobustScaler(),
        "tp_scaler": RobustScaler(),
        "u10_scaler": RobustScaler(),
        "v10_scaler": RobustScaler(),
    }
    # fit the scalers on the training data and transform all datasets
    for scaler_name, scaler in scalers.items():
        var_name = scaler_name.split("_")[0]
        columns_to_scale = [col for col in X_train.columns if var_name in col]
        # fit the scaler on the training data by merging the variables
        all_vars = np.stack(
            [X_red[c] for c in X_red.columns if var_name in c], axis=0
        ).reshape(-1, 1)
        # print(f"fit {var_name} on {all_vars.shape=}")
        scaler.fit(all_vars)
        # transform each column of the dataframe
        for col_name in columns_to_scale:
            X_train_norm.loc[:, [col_name]] = scaler.transform(
                X_train.loc[:, [col_name]].values
            )
            X_test_norm.loc[:, [col_name]] = scaler.transform(
                X_test.loc[:, [col_name]].values
            )
            X_val_norm.loc[:, [col_name]] = scaler.transform(
                X_val.loc[:, [col_name]].values
            )
        # Transform the columns from Paris
        target_column = "paris" + "_" + var_name
        Y_train_norm.loc[:, [target_column]] = scaler.transform(
            Y_train.loc[:, [target_column]].values
        )
        Y_test_norm.loc[:, [target_column]] = scaler.transform(
            Y_test.loc[:, [target_column]].values
        )
        Y_val_norm.loc[:, [target_column]] = scaler.transform(
            Y_val.loc[:, [target_column]].values
        )

    return (
        X_train_norm,
        X_test_norm,
        X_val_norm,
        Y_train_norm,
        Y_test_norm,
        Y_val_norm,
        scalers,
    )


# Deprecated
def load_normalize_data():
    """Load the previous prepared normalized dataset
    returns: X_train, X_test, y_train, Y_test"""
    # Load variable_datasets_X_train from HDF5
    variable_datasets_X_train = {}
    with pd.HDFStore("data/norm_data/variable_datasets_X_train.h5") as store:
        for variable in store.keys():
            variable_name = variable.strip("/")
            variable_datasets_X_train[variable_name] = store[variable]

    # Load variable_datasets_X_test from HDF5
    variable_datasets_X_test = {}
    with pd.HDFStore("data/norm_data/variable_datasets_X_test.h5") as store:
        for variable in store.keys():
            variable_name = variable.strip("/")
            variable_datasets_X_test[variable_name] = store[variable]

    # Load variable_datasets_y_train from HDF5
    variable_datasets_y_train = {}
    with pd.HDFStore("data/norm_data/variable_datasets_y_train.h5") as store:
        for variable in store.keys():
            variable_name = variable.strip("/")
            variable_datasets_y_train[variable_name] = store[variable]

    # Load variable_datasets_y_test from HDF5
    variable_datasets_y_test = {}
    with pd.HDFStore("data/norm_data/variable_datasets_y_test.h5") as store:
        for variable in store.keys():
            variable_name = variable.strip("/")
            variable_datasets_y_test[variable_name] = store[variable]

    return (
        variable_datasets_X_train,
        variable_datasets_X_test,
        variable_datasets_y_train,
        variable_datasets_y_test,
    )


def load_red_data():
    """Load the previous prepared reduced dataset and split it into train, test and validation sets.
    returns: X_train, X_test, X_val, y_train, y_test, y_val"""
    # Read the data
    X_red = pd.read_csv("data/red_data/X_red.csv", index_col=0, parse_dates=True)
    Y_red = pd.read_csv("data/red_data/Y_red.csv", index_col=0, parse_dates=True)
    # Split the data into train, test and validation sets
    end_train_date = "2016-01-01"
    end_test_date = "2019-01-01"
    X_train, X_test, X_val = split_train_test_val(X_red, end_train_date, end_test_date)
    Y_train, Y_test, Y_val = split_train_test_val(Y_red, end_train_date, end_test_date)

    return X_train, X_test, X_val, Y_train, Y_test, Y_val
