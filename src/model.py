#!/usr/bin/env python3

"""Implements a machine learning model for the Ames Housing dataset."""

import yaml

import lightgbm as lgb
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split

from .utils import PROBLEMATIC_COLUMNS_PATH, COLUMN_ENCODINGS_PATH


class AmesRegressor(BaseEstimator, RegressorMixin):
    """Regressor for the Ames Housing dataset.

    Attributes
    ----------
        random_state: Random state for the split and training. Defaults to None.
        model: LightGBM model for implementing regression.
    """

    with open(PROBLEMATIC_COLUMNS_PATH, "r", encoding="utf-8") as f:
        _problematic_columns = yaml.safe_load(f)

    with open(COLUMN_ENCODINGS_PATH, "r", encoding="utf-8") as f:
        _ordinal_encoders = yaml.safe_load(f)

    def __init__(self, random_state: int = None) -> None:
        self.random_state = random_state

        self.model = lgb.LGBMRegressor(
            random_state=self.random_state,
            n_estimators=20000,
            learning_rate=0.01,
            num_leaves=20,
            max_depth=3,
            reg_alpha=1e-3,
            reg_lambda=1e-2,
            subsample=0.8,
            subsample_freq=1,
            n_jobs=-1,
            verbosity=-1,
        )

    def _encode_x(self, x: pd.DataFrame) -> pd.DataFrame:
        x_c = x.copy()

        for col, enc in self._ordinal_encoders.items():
            if col in x_c:
                x_c[col] = x_c[col].map(enc).astype(np.float32).fillna(0.0)

        return x_c

    def fit(self, x: pd.DataFrame, y: pd.Series):
        """Fits the model on the provided data.

        Parameters
        ----------
            x: Pandas DataFrame containing the features.
            y: Pandas Series containing the target.
        """
        x = x.reindex(sorted(x.columns), axis=1)

        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=0.25, random_state=self.random_state
        )

        x_train = self._encode_x(x_train).drop(
            columns=self._problematic_columns, errors="ignore"
        )
        x_val = self._encode_x(x_val).drop(
            columns=self._problematic_columns, errors="ignore"
        )

        self.model.fit(
            x_train,
            y_train,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            callbacks=[lgb.early_stopping(2000, verbose=0)],
        )

        return self

    def predict(self, x: pd.DataFrame) -> pd.Series:
        """Generates predictions for the given data.

        Parameters
        ----------
            x: Pandas DataFrame containing the data.

        Returns
        -------
            Pandas Series containing the predictions.
        """
        x = x.reindex(sorted(x.columns), axis=1)

        x = self._encode_x(x).drop(columns=self._problematic_columns, errors="ignore")

        y_pred = pd.Series(
            data=self.model.predict(x),
            index=x.index,
            dtype=np.float32,
            name="SalePrice",
        )

        return y_pred
