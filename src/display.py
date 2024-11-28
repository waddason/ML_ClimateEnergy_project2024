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
