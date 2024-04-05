#!/usr/bin/env python3
# pylint: disable=C0115, C0116, W0212

"""Implements unit tests for the model."""

import pickle
import unittest

import numpy as np
import pandas as pd
import yaml

from src.utils import load_ames_data, MODEL_PATH, TARGET_COL, COLUMN_ENCODINGS_PATH


class TestModel(unittest.TestCase):
    def test_predict(self):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        df = load_ames_data().iloc[:5, :]

        x = df.drop(columns=TARGET_COL)

        y_pred = model.predict(x)

        self.assertTrue(
            df.index.equals(y_pred.index), "Prediction index must match input's index"
        )

    def test_encoder(self):
        with open(COLUMN_ENCODINGS_PATH, "r", encoding="utf-8") as f:
            column_encodings = yaml.safe_load(f)

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        df = load_ames_data().iloc[:5, :]

        x = df.drop(columns=TARGET_COL)
        enc_x: pd.DataFrame = model._encode_x(x)

        dtypes = enc_x.dtypes

        for col in dtypes:
            if col in column_encodings:
                self.assertIs(
                    dtypes[col], np.float32, f"Column {col} should be encoded"
                )
