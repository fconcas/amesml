#!/usr/bin/env python3

"""Defines project constants and utils."""

import urllib.request

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pandas.api.types import CategoricalDtype

ROOT_DIR = Path(__file__).parent.parent

DATA_URL = "https://jse.amstat.org/v19n3/decock/AmesHousing.txt"
DATA_DIR = ROOT_DIR / "data"
DATA_PATH = DATA_DIR / "AmesHousing.txt"

MODEL_DIR = ROOT_DIR / "model"
MODEL_PATH = MODEL_DIR / "ames_regressor.pickle"

CONFIG_DIR = ROOT_DIR / "config"
COLUMN_ENCODINGS_PATH = CONFIG_DIR / "column_encodings.yaml"
COLUMN_GROUPS_PATH = CONFIG_DIR / "column_groups.yaml"
COLUMN_TYPES_PATH = CONFIG_DIR / "column_types.yaml"
DATASET_CONFIG_PATH = CONFIG_DIR / "dataset_config.yaml"
GUI_DICTIONARY_PATH = CONFIG_DIR / "gui_dictionary.yaml"
GUI_GROUPS_PATH = CONFIG_DIR / "gui_groups.yaml"
PROBLEMATIC_COLUMNS_PATH = CONFIG_DIR / "problematic_columns.yaml"

TARGET_COL = "SalePrice"


def load_dtypes() -> dict:
    """Loads and formats the data types.

    Returns
    -------
        Dictionary containing the data types.
    """
    with open(COLUMN_TYPES_PATH, "r", encoding="utf-8") as f:
        column_types: dict = yaml.safe_load(f)

    # Prepares the data types using the proper format, as they
    # are not encoded by YAML files.
    for col, t in column_types.items():
        column_types[col] = np.float32 if t == "numerical" else CategoricalDtype(**t)

    return column_types


def load_gui_cols() -> dict:
    """Helper procedure to load columns formatted for the GUI.

    Returns
    -------
        Dictionary containing column types.
    """
    with open(COLUMN_TYPES_PATH, "r", encoding="utf-8") as f:
        column_types: dict = yaml.safe_load(f)

    # Assigns a 0 or the categories to the col key in the dictionary.
    # 0 is used as a placeholder for a non-iterable variable in Jinja.
    for col, t in column_types.items():
        column_types[col] = 0 if t == "numerical" else t["categories"]

    # Months are normally encoded as integers, this makes the interface
    # interpret them as a category.
    column_types["Mo Sold"] = [f"{n+1:02}" for n in range(12)]

    with open(PROBLEMATIC_COLUMNS_PATH, "r", encoding="utf-8") as f:
        problematic_columns: list = yaml.safe_load(f)

    with open(GUI_GROUPS_PATH, "r", encoding="utf-8") as f:
        gui_groups: dict = yaml.safe_load(f)

    # Assign the formatted categories to the GUI groups, if the corresponding
    # column is not found in the problematic columns.
    for group, cols in gui_groups.items():
        gui_groups[group] = {
            col: column_types[col] for col in cols if col not in problematic_columns
        }

    return gui_groups


def load_ames_data() -> pd.DataFrame:
    """Loads the Ames Housing dataset.

    Returns
    -------
        Pandas dataframe containing the data.
    """
    # Create data directory if it doesn't exist
    if not DATA_DIR.is_dir():
        DATA_DIR.mkdir()

    if not DATA_PATH.is_file():
        print("Data not found. Downloading it...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)

    dtypes = load_dtypes()

    print("Loading the data...")

    df = pd.read_csv(
        DATA_PATH,
        index_col="Order",
        sep="\t",
        dtype=dtypes,
    ).drop(columns="PID")

    return df


def format_ames_data(df: pd.DataFrame) -> pd.DataFrame:
    """Formats a Pandas DataFrame.

    Used when the data is not directly loaded from a CSV file.
    For example, when the data is passed from a web interface.

    Parameters
    ----------
        df: Pandas dataframe containing the data.

    Returns
    -------
        Pandas dataframe containing the formatted data.
    """
    dtypes = load_dtypes()

    for col in df:
        df[col] = df[col].astype(dtypes[col])

    return df
