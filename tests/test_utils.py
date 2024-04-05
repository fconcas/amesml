#!/usr/bin/env python3
# pylint: disable=C0115, C0116, W0212

"""Implements unit tests for the `utils` module."""

from pathlib import Path

import unittest

import numpy as np
import yaml

from pandas import CategoricalDtype

from src.utils import (
    load_dtypes,
    load_gui_cols,
    ROOT_DIR,
    GUI_GROUPS_PATH,
    PROBLEMATIC_COLUMNS_PATH,
    COLUMN_TYPES_PATH,
    MODEL_DIR,
    MODEL_PATH,
    CONFIG_DIR,
    DATA_DIR,
    COLUMN_ENCODINGS_PATH,
    COLUMN_GROUPS_PATH,
    DATASET_CONFIG_PATH,
    GUI_DICTIONARY_PATH,
    TARGET_COL,
)

DATA_COLS = [
    "Order",
    "PID",
    "MS SubClass",
    "MS Zoning",
    "Lot Frontage",
    "Lot Area",
    "Street",
    "Alley",
    "Lot Shape",
    "Land Contour",
    "Utilities",
    "Lot Config",
    "Land Slope",
    "Neighborhood",
    "Condition 1",
    "Condition 2",
    "Bldg Type",
    "House Style",
    "Overall Qual",
    "Overall Cond",
    "Year Built",
    "Year Remod/Add",
    "Roof Style",
    "Roof Matl",
    "Exterior 1st",
    "Exterior 2nd",
    "Mas Vnr Type",
    "Mas Vnr Area",
    "Exter Qual",
    "Exter Cond",
    "Foundation",
    "Bsmt Qual",
    "Bsmt Cond",
    "Bsmt Exposure",
    "BsmtFin Type 1",
    "BsmtFin SF 1",
    "BsmtFin Type 2",
    "BsmtFin SF 2",
    "Bsmt Unf SF",
    "Total Bsmt SF",
    "Heating",
    "Heating QC",
    "Central Air",
    "Electrical",
    "1st Flr SF",
    "2nd Flr SF",
    "Low Qual Fin SF",
    "Gr Liv Area",
    "Bsmt Full Bath",
    "Bsmt Half Bath",
    "Full Bath",
    "Half Bath",
    "Bedroom AbvGr",
    "Kitchen AbvGr",
    "Kitchen Qual",
    "TotRms AbvGrd",
    "Functional",
    "Fireplaces",
    "Fireplace Qu",
    "Garage Type",
    "Garage Yr Blt",
    "Garage Finish",
    "Garage Cars",
    "Garage Area",
    "Garage Qual",
    "Garage Cond",
    "Paved Drive",
    "Wood Deck SF",
    "Open Porch SF",
    "Enclosed Porch",
    "3Ssn Porch",
    "Screen Porch",
    "Pool Area",
    "Pool QC",
    "Fence",
    "Misc Feature",
    "Misc Val",
    "Mo Sold",
    "Yr Sold",
    "Sale Type",
    "Sale Condition",
    "SalePrice",
]


class TestUtils(unittest.TestCase):
    def test_constants(self):
        self.assertTrue(ROOT_DIR.is_dir())
        self.assertTrue(ROOT_DIR.is_absolute())
        self.assertEqual(ROOT_DIR, Path(__file__).parent.parent)

        self.assertTrue(DATA_DIR.is_dir())

        self.assertTrue(MODEL_DIR.is_dir())
        self.assertTrue(MODEL_PATH.is_file())

        self.assertTrue(CONFIG_DIR.is_dir())
        self.assertTrue(COLUMN_ENCODINGS_PATH.is_file())
        self.assertTrue(COLUMN_GROUPS_PATH.is_file())
        self.assertTrue(COLUMN_TYPES_PATH.is_file())
        self.assertTrue(DATASET_CONFIG_PATH.is_file())
        self.assertTrue(GUI_DICTIONARY_PATH.is_file())
        self.assertTrue(GUI_GROUPS_PATH.is_file())
        self.assertTrue(PROBLEMATIC_COLUMNS_PATH.is_file())

        self.assertIn(TARGET_COL, DATA_COLS)

    def test_load_dtypes(self):
        with open(COLUMN_TYPES_PATH, "r", encoding="utf-8") as f:
            column_types: dict = yaml.safe_load(f)

        dtypes = load_dtypes()

        for col, dtype in dtypes.items():
            self.assertIn(col, DATA_COLS, f"Column {col} does not exist in the dataset")
            self.assertIn(
                col, column_types, f"Column {col} not found in file {COLUMN_TYPES_PATH}"
            )

            if column_types[col] == "numerical":
                self.assertIs(
                    dtype, np.float32, f"Column {col} should be a 32-bit float"
                )
            else:
                self.assertEqual(
                    dtype,
                    CategoricalDtype(**column_types[col]),
                    f"Column {col} categories not encoded properly",
                )

    def test_load_gui_cols(self):
        with open(COLUMN_TYPES_PATH, "r", encoding="utf-8") as f:
            column_types: dict = yaml.safe_load(f)

        with open(PROBLEMATIC_COLUMNS_PATH, "r", encoding="utf-8") as f:
            problematic_columns: list = yaml.safe_load(f)

        with open(GUI_GROUPS_PATH, "r", encoding="utf-8") as f:
            gui_groups: dict = yaml.safe_load(f)

        gui_groups = load_gui_cols()

        for group, cols in gui_groups.items():
            self.assertIn(group, gui_groups, f"Group {group} not found in ")

            for col, vals in cols.items():
                self.assertIn(
                    col, DATA_COLS, f"Column {col} does not exist in the dataset"
                )
                self.assertIn(
                    col, column_types, f"Column {col} not found in {COLUMN_TYPES_PATH}"
                )
                self.assertNotIn(
                    col,
                    problematic_columns,
                    f"Column {col} found in {PROBLEMATIC_COLUMNS_PATH}, should be dropped",
                )

                if col == "Mo Sold" or column_types[col] != "numerical":
                    self.assertIsInstance(
                        vals, list, f"Column {col} should contain a list of values"
                    )
                else:
                    self.assertEqual(vals, 0, f"Column {col} should contain a 0")


if __name__ == "__main__":
    unittest.main()
