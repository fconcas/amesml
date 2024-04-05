#!/usr/bin/env python3

"""Implements a web interface for the ML models."""

import pickle
import sys

import argparse
import numpy as np
import pandas as pd
import yaml

from flask import Flask, render_template, request, jsonify
from waitress import serve

from src.utils import GUI_DICTIONARY_PATH, MODEL_PATH, format_ames_data, load_gui_cols


app = Flask(__name__)


@app.route("/")
def index():
    """Produces the index page.

    Returns
    -------
        String containing the render template.
    """
    return render_template(
        "index.html", gui_cols=app.config["gui_cols"], namings=app.config["namings"]
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Returns the prediction used by the index page.

    Returns
    -------
        Response prediction in JSON format.
    """
    response = pd.Series(request.form.to_dict())
    response = response.map(lambda x: np.nan if x == "" else x)
    response = pd.DataFrame([response])
    response = format_ames_data(response)

    y_pred = app.config["model"].predict(response)[0]

    return jsonify({"pred": str(y_pred)})


def parse_arguments():
    """Parses the command line arguments.
    
    Returns
    -------
        Namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--address", default="127.0.0.1")
    parser.add_argument("-p", "--port", default=5000)
    args = parser.parse_args()

    return args


def main():
    """Main function."""
    args = parse_arguments()

    if not MODEL_PATH.exists():
        sys.exit("Model not found. Train one with the script train_model.py")

    with open(MODEL_PATH, "rb") as f:
        app.config["model"] = pickle.load(f)

    with open(GUI_DICTIONARY_PATH, "r", encoding="utf-8") as f:
        app.config["namings"] = yaml.safe_load(f)

    app.config["gui_cols"] = load_gui_cols()

    print("Server running...")
    serve(app, host=args.address, port=args.port)


if __name__ == "__main__":
    main()
