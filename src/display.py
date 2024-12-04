"""
display.py

This module contains utility functions for the ML_ClimateEnergy_project2024.

Author: Tristan Waddington
Date: nov 2024
"""

###############################################################################
# Imports
###############################################################################
from pathlib import Path

# cartopy to display maps
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import mean_squared_error
import src.load_data as ld
from importlib import reload

reload(ld)

###############################################################################
# Constants
###############################################################################

cities_latlon = {
    "paris": (48.75, 2.25),
    "marseille": (43.25, 5.5),
    "brest": (48.5, 355.5),
    "london": (51.5, 359.75),
    "berlin": (52.5, 13.5),
}

# Variables explanation
var_legend = {
    "t2m": "Air temperature at 2 m above the ground [K]",
    "d2m": "Dew point at 2 m above the ground [K]",
    "u10": "Zonal wind component at 10 m [m/s]",
    "v10": "Meridional wind component at 10 m [m/s]",
    "skt": "Skin temperature [K]",
    "tcc": "Total cloud cover [0-1]",
    "sp": "Surface pressure [Pa]",
    "tp": "Total precipitation [m]",
    "ssrd": "Surface solar radiation (downwards) [J/m^2]",
    "blh": "Boundary layer height [m]",
}

# The study of cycles was done by hand
# Seasonal cycles
has_seasonal_cycle = {
    "t2m": True,
    "d2m": True,
    "u10": False,
    "v10": False,
    "skt": True,
    "tcc": False,
    "sp": False,
    "tp": False,
    "ssrd": True,
    "blh": True,
}
has_daily_cycle = {
    "t2m": True,
    "d2m": True,
    "u10": True,
    "v10": True,
    "skt": True,
    "tcc": False,
    "sp": True,
    "tp": False,
    "ssrd": True,
    "blh": True,
}


###############################################################################
# Cartography functions
###############################################################################
def display_weather_stations():
    """Plot the weather station on a map of Europe"""
    # Load coordinates from other notebook

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([-15, 25, 35, 65], crs=ccrs.PlateCarree())

    # Draw the background
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    # ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)
    ax.gridlines(draw_labels=True)

    # Add cities' names and positions
    for city, (lat, lon) in cities_latlon.items():
        ax.text(
            lon + 0.5,
            lat + 0.5,
            city.capitalize(),
            bbox=dict(facecolor="white", alpha=0.5),
            color="b",
            transform=ccrs.PlateCarree(),
        )
        ax.plot(lon, lat, "bo", transform=ccrs.PlateCarree())

    ax.set_title("Weather Stations in Europe")

    plt.show()


###############################################################################
# Display weather features
###############################################################################
def display_weather_features(df: pd.DataFrame, city: str):
    """Display the weather features for a given city.
    df: DataFrame containing the weather features.
    return 3 plot of the weather features:
    - Temperature: Air temperature, Dew point, Skin temperature [K]
    - Precipitation: Rainfall [m] and surface pressure [Pa]
    - Could and Wind: Cloud cover [%], Wind speed [m/s]"""
    # Display the evolution of some data
    fig, axs = plt.subplots(3, 1, figsize=(15, 10), layout="constrained")
    fig.suptitle(f"Evolution of weather data in {city.capitalize()}")

    # Plot the temperatures
    axs[0].plot(df.index, df["t2m"], label=var_legend["t2m"])
    axs[0].plot(df.index, df["d2m"], label=var_legend["d2m"])
    axs[0].plot(df.index, df["skt"], label=var_legend["skt"])
    axs[0].set_title("Temperature evolution")
    axs[0].set_ylabel("Temperature [K]")
    axs[0].legend()

    # Plot the precipitation, cloud cover and pressure
    axs[1].plot(df.index, df["tp"], c="b", label=var_legend["tp"])
    axs[1].set_ylabel("Total precipitation", c="b")
    ax_twin1 = axs[1].twinx()
    ax_twin1.plot(df.index, df["sp"], c="g", label=var_legend["sp"])
    ax_twin1.set_ylabel("Surface pressure", c="g")
    axs[1].set_title("Precipitation and surface pressure evolution")
    axs[1].legend(loc="upper left")
    ax_twin1.legend(loc="upper right")

    # Plot the wind components and cloud cover
    axs[2].plot(df.index, df["u10"], label=var_legend["u10"])
    axs[2].plot(df.index, df["v10"], label=var_legend["v10"])
    axs[2].set_title("Wind components and cloud cover evolution ")
    axs[2].set_ylabel("Wind component [m/s]")
    axs[2].legend(loc="upper left")
    ax_twin2 = axs[2].twinx()
    # Skrink down the cloud cover to display
    ax_twin2.set_ylim(0, 3)
    ax_twin2.plot(df.index, df["tcc"], c="0.3", label=var_legend["tcc"])
    ax_twin2.legend(loc="upper right")
    ax_twin2.set_ylabel("Total cloud cover (shrinked)")

    plt.show()


###############################################################################
# Display cycles
###############################################################################
def display_cycles(df, start_year: int, end_year: int):
    """Display the seasonal and daily cycles of the weather features
    df: DataFrame containing the weather features.
    start_year: int, start year of the data
    end_year: int, end year of the data (included)"""
    # Display the seasonal and daily cycles
    fig, axs = plt.subplots(10, 2, figsize=(15, 25), layout="constrained")
    fig.suptitle(f"Seasonal and daily cycles of weather data {start_year}-{end_year}")

    for year in range(start_year, end_year + 1):
        # sub df by year
        df_year = df.loc[df.index.year == year]
        # df_monthly = df_year.groupby(df_year.index.dayofyear).mean()
        df_monthly = df_year.groupby(df_year.index.month).mean()
        df_hourly = df_year.groupby(df_year.index.hour).mean()
        for ax_i, var_name in enumerate(df.columns):
            # Plot the seasonal cycles on the left
            df_monthly[var_name].plot(ax=axs[ax_i, 0], label=year, linewidth=0.7)
            axs[ax_i, 0].set_title(f"Seasonal cycles of {var_name}")
            axs[ax_i, 0].set_xlabel("Month of the year")
            axs[ax_i, 0].legend()
            if not has_seasonal_cycle.get(var_name):
                axs[ax_i, 0].set_facecolor("lightgrey")

            # Plot the daily cycles on the right
            df_hourly[var_name].plot(ax=axs[ax_i, 1], label=year, linewidth=0.7)
            axs[ax_i, 1].set_title(f"Daily cycles of {var_name}")
            axs[ax_i, 1].set_xlabel("Hour of the day")
            axs[ax_i, 1].legend()
            if not has_daily_cycle.get(var_name):
                axs[ax_i, 1].set_facecolor("lightgrey")

    plt.show()


###############################################################################
# Display MSE
###############################################################################
def display_mse_by_year(
    y_pred: pd.DataFrame, y_true: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame.style]:
    """Display the Mean Squared Error by year
    Return the dataframe and the colored dataframe for standardized display"""
    # compute the mse by variable and by year, then the mean
    mse_per_var = {}
    for i_var, variable in enumerate(y_true.columns):
        mse_per_var[variable] = []
        for year in [2016, 2017, 2018]:
            mse = mean_squared_error(
                y_true[variable].iloc[y_true.index.year == year],
                y_pred.iloc[y_true.index.year == year, i_var],
            )
            mse_per_var[variable].append(mse)
        # compute the mean for the whole dataset
        mse_per_var[variable].append(np.mean(mse_per_var[variable]))
    # load into a dataframe and display with a color gradient
    mse_by_year_df = pd.DataFrame(mse_per_var, index=[2016, 2017, 2018, "all"])
    style_df = mse_by_year_df.style.background_gradient(cmap="RdYlGn_r")
    return mse_by_year_df, style_df


def display_delta(
    mse_ref: pd.DataFrame, mse_new: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame.style]:
    """Display the delta between two dataframes
    Return the dataframe and the colored dataframe for standardized display"""
    delta = mse_ref - mse_new
    v_min = delta.values.min()
    v_max = delta.values.max()
    v_abs = max(abs(v_min), abs(v_max))
    style_df = delta.style.background_gradient(cmap="RdYlGn", vmin=-v_abs, vmax=v_abs)
    return delta, style_df


###############################################################################
# Display losses during DL training
###############################################################################
def display_losses(
    train_losses: list[float], test_losses: list[float], model_name: str = None
) -> plt:
    """Display the training and test losses during the training of a model"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Training loss")
    ax.plot(test_losses, label="Test loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(
        f"Training and test losses of {model_name}"
        if model_name
        else "Training and test losses"
    )
    # display the min test loss
    i_min_loss = test_losses.index(min(test_losses))
    ax.plot(i_min_loss, test_losses[i_min_loss], "ro", label="Min test loss")

    ax.legend()
    plt.show()


##########################
# Display the predictions
###########################"
def display_predictions_with_actual(
    y_pred: pd.DataFrame,
    y_true: pd.DataFrame,
    scalers: dict[str:any],
    from_date=None,
    to_date=None,
    var_list: list[str] = None,
) -> None:
    """Display the predictions and the actual values for the given time frame grouped by corresponding vars.
    Values are scaled back to the actual units.
    Return plots depending on given vars:
    - Temperature: Air temperature, Dew point, Skin temperature [K]
    - Precipitation: Rainfall [m] and surface pressure [Pa]
    - Could and Wind: Cloud cover [%], Wind speed [m/s]
    - Boundary layer height and solar radiation [m] and [J/m^2]

    y_pred: DataFrame containing the predictions
    y_true: DataFrame containing the actual values
    scalers: dict of scalers used to scale back the data
    from_date: str, starting date of the time frame in format 'YYYY-MM-DD'
    to_date: str, ending date of the time frame in format 'YYYY-MM-DD'
    var_list: list of columns of y_true to display, if None, display all available variables
    """
    # Variable selection and validation
    if var_list is None:
        # Display all vars if none is given
        var_list = y_true.columns
    else:
        assert all(
            var in y_true.columns for var in var_list
        ), "Some variables are not in the DataFrame y_true"
        assert all(
            var in y_pred.columns for var in var_list
        ), "Some variables are not in the DataFrame y_pred"

    predic_values = pd.DataFrame()
    real_values = pd.DataFrame()

    # date range selection and validation
    if from_date is None:
        from_date_dt = y_true.index.min()
    else:
        from_date_dt = pd.to_datetime(from_date)
    if to_date is None:
        to_date_dt = y_true.index.max()
    else:
        to_date_dt = pd.to_datetime(to_date)
    assert from_date_dt <= to_date_dt, "from_date must be before to_date"
    assert (
        y_true.index.min() <= from_date_dt <= y_true.index.max()
    ), "from_date not in y_true index"
    assert (
        y_true.index.min() <= to_date_dt <= y_true.index.max()
    ), "to_date not in y_true index"
    assert (
        y_pred.index.min() <= from_date_dt <= y_pred.index.max()
    ), "from_date not in y_pred index"
    assert (
        y_pred.index.min() <= to_date_dt <= y_pred.index.max()
    ), "to_date not in y_pred index"

    # Scale back the variables on the requested time frame and variables
    real_values = ld.scale_back_df(
        y_true.loc[from_date_dt:to_date_dt, var_list], scalers
    )
    predic_values = ld.scale_back_df(
        y_pred.loc[from_date_dt:to_date_dt, var_list], scalers
    )

    # Acutal plot
    var_to_plot = real_values.columns
    var_ax_repartition = {
        "t2m": 0,
        "d2m": 0,
        "u10": 2,
        "v10": 2,
        "skt": 0,
        "tcc": 2,
        "sp": 1,
        "tp": 1,
        "ssrd": 3,
        "blh": 3,
    }
    ax_to_plot = set(var_ax_repartition[var] for var in var_to_plot)
    nb_axs = len(ax_to_plot)
    fig, axs = plt.subplots(nb_axs, 1, figsize=(15, 5 * nb_axs), layout="constrained")
    # Put a single ax in a list
    if nb_axs == 1:
        axs = [axs]
    fig.suptitle(f"Predictions for weather data in Paris")
    real_style = {
        "linestyle": "solid",
        "linewidth": 1.5,
    }
    predic_style = {
        "linestyle": "dotted",
        "linewidth": 1.5,
    }
    # specific color per var
    var_color = {
        "t2m": "tab:blue",
        "d2m": "tab:orange",
        "u10": "tab:green",
        "v10": "tab:red",
        "skt": "tab:green",
        "tcc": "tab:gray",
        "sp": "tab:green",
        "tp": "b",
        "ssrd": "tab:olive",
        "blh": "tab:cyan",
    }

    # TODO: check the presence of var in the df
    curr_ax = 0
    # Plot the temperatures
    if 0 in ax_to_plot:
        if "t2m" in var_to_plot:
            axs[curr_ax].plot(
                real_values.index,
                real_values["t2m"],
                label=var_legend["t2m"],
                color=var_color["t2m"],
                **real_style,
            )
            axs[curr_ax].plot(
                real_values.index,
                predic_values["t2m"],
                label=var_legend["t2m"] + " (pred)",
                color=var_color["t2m"],
                **predic_style,
            )
        if "skt" in var_to_plot:
            axs[curr_ax].plot(
                real_values.index,
                real_values["skt"],
                label=var_legend["skt"],
                color=var_color["skt"],
                **real_style,
            )
            axs[curr_ax].plot(
                real_values.index,
                predic_values["skt"],
                label=var_legend["skt"] + " (pred)",
                color=var_color["skt"],
                **predic_style,
            )

        if "d2m" in var_to_plot:
            axs[curr_ax].plot(
                real_values.index,
                real_values["d2m"],
                label=var_legend["d2m"],
                color=var_color["d2m"],
                **real_style,
            )
            axs[curr_ax].plot(
                real_values.index,
                predic_values["d2m"],
                label=var_legend["d2m"] + " (pred)",
                color=var_color["d2m"],
                **predic_style,
            )

        axs[curr_ax].set_title("Temperature evolution")
        axs[curr_ax].set_ylabel("Temperature [K]")
        axs[curr_ax].legend()

        # offset the axes
        curr_ax += 1

    # Plot the precipitation, cloud cover and pressure
    if 1 in ax_to_plot:
        if "tp" in var_to_plot:
            axs[curr_ax].plot(
                real_values.index,
                real_values["tp"],
                label=var_legend["tp"],
                color=var_color["tp"],
                **real_style,
            )
            axs[curr_ax].plot(
                real_values.index,
                predic_values["tp"],
                label=var_legend["tp"] + " (pred)",
                color=var_color["tp"],
                **predic_style,
            )
            axs[curr_ax].legend(loc="upper left")
        if "sp" in var_to_plot:
            ax_twin1 = axs[curr_ax].twinx()
            ax_twin1.plot(
                real_values.index,
                real_values["sp"],
                label=var_legend["sp"],
                color=var_color["sp"],
                **real_style,
            )
            ax_twin1.plot(
                real_values.index,
                predic_values["sp"],
                label=var_legend["sp"] + " (pred)",
                color=var_color["sp"],
                **predic_style,
            )
            ax_twin1.set_ylabel("Surface pressure", c=var_color["sp"])
            ax_twin1.legend(loc="upper right")

        axs[curr_ax].set_ylabel("Total precipitation", c=var_color["tp"])
        axs[curr_ax].set_title("Precipitation and surface pressure evolution")

        # offset the axes
        curr_ax += 1

    # Plot the wind components and cloud cover
    if 2 in ax_to_plot:
        if "u10" in var_to_plot:
            axs[curr_ax].plot(
                real_values.index,
                real_values["u10"],
                label=var_legend["u10"],
                color=var_color["u10"],
                **real_style,
            )
            axs[curr_ax].plot(
                real_values.index,
                predic_values["u10"],
                label=var_legend["u10"] + " (pred)",
                color=var_color["u10"],
                **predic_style,
            )
        if "v10" in var_to_plot:
            axs[curr_ax].plot(
                real_values.index,
                real_values["v10"],
                label=var_legend["v10"],
                color=var_color["v10"],
                **real_style,
            )
            axs[curr_ax].plot(
                real_values.index,
                predic_values["v10"],
                label=var_legend["v10"] + " (pred)",
                color=var_color["v10"],
                **predic_style,
            )
            axs[curr_ax].legend(loc="upper left")

        axs[curr_ax].set_title("Wind components and cloud cover evolution ")
        axs[curr_ax].set_ylabel("Wind component [m/s]")

        # Skrink down the cloud cover to display
        if "tcc" in var_to_plot:
            ax_twin2 = axs[curr_ax].twinx()
            ax_twin2.set_ylim(0, 3)
            ax_twin2.plot(
                real_values.index,
                real_values["tcc"],
                label=var_legend["tcc"],
                color=var_color["tcc"],
                **real_style,
            )
            ax_twin2.plot(
                real_values.index,
                predic_values["tcc"],
                label=var_legend["tcc"] + " (pred)",
                color=var_color["tcc"],
                **predic_style,
            )
            ax_twin2.legend(loc="upper right")
            ax_twin2.set_ylabel("Total cloud cover (shrinked)", c=var_color["tcc"])
        # offset the axes
        curr_ax += 1

    # blh & ssrd
    if 3 in ax_to_plot:
        if "blh" in var_to_plot:
            axs[curr_ax].plot(
                real_values.index,
                real_values["blh"],
                label=var_legend["blh"],
                color=var_color["blh"],
                **real_style,
            )
            axs[curr_ax].plot(
                real_values.index,
                predic_values["blh"],
                label=var_legend["blh"] + " (pred)",
                color=var_color["blh"],
                **predic_style,
            )
            axs[curr_ax].legend(loc="upper left")
        if "ssrd" in var_to_plot:
            ax_twin3 = axs[curr_ax].twinx()
            ax_twin3.plot(
                real_values.index,
                real_values["ssrd"],
                label=var_legend["ssrd"],
                color=var_color["ssrd"],
                **real_style,
            )
            ax_twin3.plot(
                real_values.index,
                predic_values["ssrd"],
                label=var_legend["ssrd"] + " (pred)",
                color=var_color["ssrd"],
                **predic_style,
            )
            ax_twin3.legend(loc="upper right")
            ax_twin3.set_ylabel("Surface solar radiation [J/m^2]", c=var_color["ssrd"])

        axs[curr_ax].set_title("Boundary layer height and solar radiation")
        axs[curr_ax].set_ylabel("Boundary layer height [m]", c=var_color["blh"])

    plt.show()
    return
