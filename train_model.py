#!/usr/bin/env python3

"""Implements a script to train the machine learning model."""

import pickle

from src.model import AmesRegressor
from src.utils import TARGET_COL, MODEL_PATH, load_ames_data


def main():
    """The main function."""

    ames_df = load_ames_data()

    x = ames_df.drop(columns=TARGET_COL)
    y = ames_df[TARGET_COL]

    print("Training the model...")

    model = AmesRegressor()
    model.fit(x, y)

    print("Saving the model...")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("Done.")


if __name__ == "__main__":
    main()
