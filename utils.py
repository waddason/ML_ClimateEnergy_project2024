"""
utils.py

This module contains utility functions for the ML_ClimateEnergy_project2024.

Author: Tristan Waddington
Date: 2024

"""

# Import necessary libraries
import pandas as pd


# Define utility functions below


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
