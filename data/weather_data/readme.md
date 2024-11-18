# Machine Learning for climate and energy - 2024
IP-Paris - Group project

## Weather data set

This data set comes from ERA5 and contains surface data for 5 cities:

- Berlin
- Brest
- London
- Marseille
- Paris

There is 1 netcdf file per variable per city.
Each file has 40 years of data (except Paris: 41 years)

Some variables are *Analysis* variables
Some variables are *Forecast* variables and are shifted by 3h compared to analysis variable. (you need to rely on the timestamp that is in the netcdf file).

You can use the xarray library to load netcdf files in python and you can then convert this data set into a pandas DataFrame.

**The data is prepared, cleaned and normalize in the `norm_Data` folder. It can be imported thanks to the
`utils.load_normalized_data()` function.**
> Have a look in the draft notebooks.

## Get Started

Ready to contribute? Here's how to set up `MLCIMATE_ENERERGY_PROJECT_2024` for local development.

1. **Fork** the repo on GitHub.
2. **Clone** your fork locally.
3. Create a specific virtual environment named 'ML4CE' with `conda` from the `environment.yml` file in the repo:
    ```conda env create -n ML4CE --file environment.yml```
4. Create a **branch** for local development and save changes locally.
5. **Commit** your changes
6. **Push** your branch to GitHub.
7. Submit a **pull request** through the GitHub website.