# ML_ClimateEnergy_project2024
Student project for Machine Learning for Climate and Energy 2024 lesson

[Lesson ressources](https://energy4climate.pages.in2p3.fr/public/education/machine_learning_for_climate_and_energy/chapters/frontmatter.html) By Bruno Deremble and Alexis Tantet.


## Project aim

**Aim**: <br>
Paris weather station was cyber-hacked, and is therefore unable to share
its weather measures since the beginiing of 2019. **Our task is to predict this missing data from
other European weather stations measures**.

Furthermore, we suspect that since 2016 some reports have been modified by these 
hackers in a previous altering attack. We have to spot these modifications.

**Dataset**: <br>
We have access to the full history of measurements of 5 stations, including
Paris, reporting 10 weather variables that will be described bellow. 
The measures were reported hourly and span form year 1980 to 2019 (included, exact common timeframe: 1980-01-01 07:00:00 to 2019-12-31 23:00:00).

For educative purpose, we decided to split them on 3 different datasets:
- Confirmed clean timeframe (train): year 1980-2015
- Suspicious time-frame (test): year 2016-2018
- "Real time" stream (validation): year 2019

## Use
To clone the repository and install the necessary dependencies, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/waddason/ML_ClimateEnergy_project2024.git
cd ML_ClimateEnergy_project2024
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate ML_climate
```

3. Open the notebook [`Main.ipynb`](Main.ipynb), Run all cells.


4. By default, the models will be loaded from the `saved_models/` folder. If you want to retrain, set 
RETRAIN_MODELS to True on the second code cell.

```python
# Constants
# Retrain Deep Learning models, if False, load the models from disk
RETRAIN_MODELS: bool = True
```

## Project Architecture

The project is organized into the following main directories and files:

- [`data/`](data/): Contains the datasets used for training, testing, and validation. 
    - norm_data/: train/test split of normalize Pandas DataFrame (not used in final project)
    - red_data/: X/y split, resample to a daily average measure per station, used in the project.
    - weather_data/: the source Xarray files per station per feature.

    
- [`notebooks/`](notebooks/): Jupyter notebooks used for data analysis and model development during the construction of the project.
- [`src/`](src/): Python scripts for data preprocessing, training, evaluation and display.
    - [`display.py`](src/display.py): Script for data visualisation (df and plot).
    - [`DL_models.py`](scripts/DL_models.py): Script for using machine learning models in `pyTorch`.
    - [`load_data.py`](scripts/load_data.py): Script for loading and preparing the data from the `data` folder.

- [`saved_models/`](saved_models/): Directory where trained models are saved.
- [`environment.yml`](environment.yml): Conda environment configuration file. List all dependencies required to run the project, ensuring reproducibility.
- **[`Main.ipynb`](Main.ipynb): Main notebook for explanation and visualization, running the project pipeline.**
- [`README.md`](README.md): Project documentation (this file).


