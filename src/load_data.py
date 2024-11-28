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
import xarray as xr

from pathlib import Path
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


# -----------------------------------------------------------------------------
# 1. Read the nc files
# -----------------------------------------------------------------------------
def load_nc_files_by_subfolders(base_path: Path) -> dict[str, pd.DataFrame]:
    """Load weather data as a dict of DataFrame by city containing all variables."""
    dataframes: dict[str, pd.DataFrame] = {}
    for subfolder in base_path.iterdir():
        if not subfolder.is_dir():
            continue
        nc_files = list(subfolder.glob("*.nc"))
        if not nc_files:
            continue
        combined_ds = xr.open_mfdataset(nc_files, combine="by_coords")
        # Drop NaN values along the time dimension
        combined_ds = combined_ds.dropna(dim="time", how="all")
        # Keep only the time as index
        combined_ds = combined_ds.to_dataframe()
        combined_ds.reset_index(
            level=["latitude", "longitude"], drop=True, inplace=True
        )
        dataframes[subfolder.name] = combined_ds
    return dataframes


# -----------------------------------------------------------------------------
# 2. Transpose the dict from cities to variables
# -----------------------------------------------------------------------------
def transpose_df_dict_from_cities_to_variables(
    df_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Transpose the DataFrame dict from cities to variables."""
    transposed_dict: dict[str, pd.DataFrame] = {}
    for city, df in df_dict.items():
        for var in df.columns:
            if var not in transposed_dict:
                transposed_dict[var] = pd.DataFrame()
            transposed_dict[var][city] = df[var]
    return transposed_dict


# -----------------------------------------------------------------------------
# 3. Reduce to a daily frequency and merge the datasets
# -----------------------------------------------------------------------------
def create_daily_X_y_datasets(
    df_dict: dict[str, pd.DataFrame],
    target_city: str,
    start_year: int,
    end_year: int,
    save_path: Path = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce the datasets to a daily frequency and merge them into X and y datasets.
    df_dict: dict of DataFrames by city containing all variables.
    target_city: (str) city to predict.
    start_year: (int) start year of the data.
    end_year: (int) end year of the data.
    returns: (DataFrame) X_red, y_red."""

    X_red = pd.DataFrame()
    y_red = pd.DataFrame()
    # Reduce the datasets to a daily frequency
    for city, df_city in df_dict.items():
        df = df_city.loc[
            (df_city.index.year >= start_year) & (df_city.index.year <= end_year)
        ]
        # resample the data to daily frequency
        daily_df = df.resample("D").mean()
        # Overwrite the columns that makes sense to be summed over the day
        daily_df["tp"] = df["tp"].resample("D").sum()
        daily_df["ssrd"] = df["ssrd"].resample("D").sum()

        # Replace NaN values with the previous value
        daily_df.ffill()

        # Create the dataset with explicit column names "city_var"
        col_names_city = [f"{city}_{var}" for var in daily_df.columns]
        daily_df.columns = col_names_city

        if city == target_city:
            y_red = daily_df
        else:
            X_red = pd.concat([X_red, daily_df], axis=1)

    # Save the reduced datasets
    if save_path is not None:
        X_red.to_csv(save_path / "X_red.csv")
        y_red.to_csv(save_path / "Y_red.csv")

    return X_red, y_red


# -----------------------------------------------------------------------------
# 4. Split the data into train, test and validation sets
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# 4. Split the data into train, test and validation sets and normalize
# -----------------------------------------------------------------------------
def load_red_norm_data(
    save_path: Path, end_train_date: str, end_test_date: str
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, RobustScaler],
]:
    """Load the previous prepared reduced dataset and split it into train, test and validation sets.
    Then normalize the data.
    save_path: (Path) path to the directory containing the reduced datasets X_red.csf and Y_red.csv.
    end_train_date: (str) last date of the train set in format 'YYYY-MM-DD' excluded.
    end_test_date: (str) last date of the test set in format 'YYYY-MM-DD' excluded.
    returns: X_train, X_test, X_val, y_train, y_test, y_val, scalers"""
    # Read the data
    assert save_path.is_dir(), f"The save path {save_path} should be a directory"
    assert (
        save_path / "X_red.csv"
    ).exists(), f"The X_red.csv file should exist in {save_path}"
    assert (
        save_path / "Y_red.csv"
    ).exists(), f"The Y_red.csv file should exist in {save_path}"
    X_red = pd.read_csv(save_path / "X_red.csv", index_col=0, parse_dates=True)
    Y_red = pd.read_csv(save_path / "Y_red.csv", index_col=0, parse_dates=True)

    # Split the data into train, test and validation sets
    X_train_norm, X_test_norm, X_val_norm = split_train_test_val(
        X_red, end_train_date, end_test_date
    )
    Y_train_norm, Y_test_norm, Y_val_norm = split_train_test_val(
        Y_red, end_train_date, end_test_date
    )

    # Normalize the data by identic features, use explicit robust scalers to avoid outliers
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
        columns_to_scale = [col for col in X_train_norm.columns if var_name in col]
        # fit the scaler on the training data by merging the variables
        all_vars = np.stack(
            [X_red[c] for c in X_red.columns if var_name in c], axis=0
        ).reshape(-1, 1)
        # print(f"fit {var_name} on {all_vars.shape=}")
        scaler.fit(all_vars)
        # transform each column of the dataframe
        for col_name in columns_to_scale:
            X_train_norm.loc[:, col_name] = pd.Series(
                scaler.transform(X_train_norm[[col_name]].values).flatten(),
                index=X_train_norm.index,
                dtype="float64",
            )
            X_test_norm.loc[:, col_name] = pd.Series(
                scaler.transform(X_test_norm[[col_name]].values).flatten(),
                index=X_test_norm.index,
                dtype="float64",
            )
            X_val_norm.loc[:, col_name] = pd.Series(
                scaler.transform(X_val_norm[[col_name]].values).flatten(),
                index=X_val_norm.index,
                dtype="float64",
            )

        # Transform the columns from Paris
        target_column = "paris" + "_" + var_name
        Y_train_norm.loc[:, target_column] = pd.Series(
            scaler.transform(Y_train_norm[[target_column]].values).flatten(),
            index=Y_train_norm.index,
            dtype="float64",
        )
        Y_test_norm.loc[:, target_column] = pd.Series(
            scaler.transform(Y_test_norm[[target_column]].values).flatten(),
            index=Y_test_norm.index,
            dtype="float64",
        )
        Y_val_norm.loc[:, target_column] = pd.Series(
            scaler.transform(Y_val_norm[[target_column]].values).flatten(),
            index=Y_val_norm.index,
            dtype="float64",
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
