"""
Backend - loads the trained model and makes predictions.

This module contains no user-interface code (no Streamlit), so it could also be
used from a script, a notebook or a web API. The frontend (frontend/app.py) calls it.

  load_bundle()          the saved Pipeline (preprocessor + model) and its metadata
  predict()              predicted class + attack probability for raw feature rows
  training_defaults()    training median / most common value of every input feature
  standardised_values()  how unusual a connection is (z-score per numeric feature)
  load_csv(), load_json() read result files written by the ML stages

Example:
    from backend import predictor
    bundle = predictor.load_bundle()
    rows = predictor.load_csv("test_samples.csv")
    print(predictor.predict(bundle, rows).head())
"""

import json

import joblib
import pandas as pd

from ml.preprocessing import ARTIFACTS_DIR

MODEL_PATH = ARTIFACTS_DIR / "nids_model.joblib"  # written by: python -m ml.classical_models


def model_available():
    return MODEL_PATH.exists()


def load_bundle(path=MODEL_PATH):
    """
    Dict with the fitted Pipeline ("pipeline") and what the frontend needs to build inputs:
    selected_features, categorical_features, categories, class_names, model_name, dataset,
    test_metrics.
    """
    return joblib.load(path)


def load_csv(name):
    """Read a CSV from resources/artifacts/, or None if it does not exist yet."""
    path = ARTIFACTS_DIR / name
    return pd.read_csv(path) if path.exists() else None


def load_json(name):
    """Read a JSON file from resources/artifacts/ (e.g. "results/svm.json"), or None."""
    path = ARTIFACTS_DIR / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def predict(bundle, rows):
    """
    rows: DataFrame with (at least) the selected feature columns, raw values.
    Returns a DataFrame with the predicted class and the probability that each row is an attack.
    """
    pipeline = bundle["pipeline"]
    proba = pipeline.predict_proba(rows[bundle["selected_features"]])
    names = [bundle["class_names"][c] for c in pipeline.classes_]
    normal_col = names.index("Normal")
    return pd.DataFrame({
        "prediction": [names[i] for i in proba.argmax(axis=1)],
        "attack_probability": 1 - proba[:, normal_col],
    }, index=rows.index)


def training_defaults(bundle):
    """Training median (numeric) / most frequent value (categorical), used to pre-fill inputs."""
    preprocessor = bundle["pipeline"].named_steps["preprocess"]
    defaults = {}
    for name, transformer, columns in preprocessor.transformers_:
        if name in ("num", "cat"):
            defaults.update(zip(columns, transformer.named_steps["impute"].statistics_))
    return defaults


def standardised_values(bundle, row):
    """
    How far each numeric feature of `row` is from the training average, in standard
    deviations (z-score), using the mean/std stored in the fitted scaler.
    """
    numeric_step = bundle["pipeline"].named_steps["preprocess"].named_transformers_["num"]
    columns = [f for f in bundle["selected_features"] if f not in bundle["categorical_features"]]
    z = numeric_step.transform(row[columns])[0]
    return pd.DataFrame({"feature": columns, "value": row[columns].iloc[0].to_numpy(), "z": z})
