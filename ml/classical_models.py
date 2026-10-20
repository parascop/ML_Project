"""
Stage 2 - Classical machine-learning models (Scikit-learn).

Trains Random Forest, Logistic Regression, SVM and Gradient Boosting on the
train/test split saved by Phase 1, so every model sees exactly the same data.
For each model it prints and saves the metrics, a confusion matrix and the model.

Run from the project root (after Phase 1):
    python -m ml.classical_models
    python -m ml.classical_models --extra-test ml/data/nsl-kdd/KDDTest+.txt   # NSL-KDD only

At the end, the chosen model (default: Gradient Boosting) is saved together with the
Phase 1 preprocessor as ONE scikit-learn Pipeline in resources/artifacts/nids_model.joblib.
The Streamlit app loads that file, so raw input is always transformed exactly as in training.
"""

import argparse
import json
import time
from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC

from ml.evaluation import compare_models, evaluate_model, load_extra_test, save_curves
from ml.preprocessing import ARTIFACTS_DIR, RANDOM_STATE, load_processed_data, to_float32


def build_models(class_weights=None):
    """
    class_weights is None when Phase 1 used SMOTE (data already balanced) and a
    {class: weight} dict when Phase 1 chose class weights instead.
    """
    cw = {int(k): v for k, v in class_weights.items()} if class_weights else None
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=100, class_weight=cw, n_jobs=-1, random_state=RANDOM_STATE),
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight=cw),
        # RBF-kernel SVM; its training time grows roughly with rows squared,
        # so it is trained on a stratified subsample (see --svm-max-rows)
        "svm": SVC(kernel="rbf", C=1.0, gamma="scale", class_weight=cw),
        # Scikit-learn's fast histogram-based gradient boosting
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200, class_weight=cw, random_state=RANDOM_STATE),
    }


def subsample(X, y, max_rows):
    if not max_rows or len(y) <= max_rows:
        return X, y
    X_small, _, y_small, _ = train_test_split(
        X, y, train_size=max_rows, stratify=y, random_state=RANDOM_STATE)
    return X_small, y_small


def save_deployment_model(model_name, artifacts_dir=ARTIFACTS_DIR):
    """Combine the fitted preprocessor and a trained model into one Pipeline for the app."""
    out = Path(artifacts_dir)
    with open(out / "metadata.json") as f:
        meta = json.load(f)
    with open(out / "results" / f"{model_name}.json") as f:
        results = json.load(f)

    pipeline = Pipeline([
        ("preprocess", joblib.load(out / "preprocessor.joblib")),  # impute, scale, one-hot
        ("to_float32", FunctionTransformer(to_float32)),            # same dtype as in training
        ("model", joblib.load(out / "models" / f"{model_name}.joblib")),
    ])
    bundle = {
        "pipeline": pipeline,
        "model_name": model_name,
        "dataset": meta["dataset"],
        "class_names": meta["class_names"],
        "selected_features": meta["selected_features"],
        "categorical_features": meta["categorical_features"],
        "categories": meta["categories"],
        "test_metrics": {k: results[k] for k in ["accuracy", "precision", "recall", "f1",
                                                 "false_positive_rate"] if k in results},
    }
    joblib.dump(bundle, out / "nids_model.joblib")
    print(f"\nSaved {model_name} + preprocessor as one pipeline: {out / 'nids_model.joblib'}")


def train_all(artifacts_dir=ARTIFACTS_DIR, only=None, svm_max_rows=20_000, extra_test=None,
              deploy_model="gradient_boosting"):
    out = Path(artifacts_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test, meta = load_processed_data(artifacts_dir)
    print(f"Train: {X_train.shape}, test: {X_test.shape}, classes: {meta['class_names']}, "
          f"balancing: {meta['balance_method']}")
    # optional second, harder test set recorded separately (e.g. KDDTest+)
    extra = load_extra_test(extra_test, meta, artifacts_dir) if extra_test else None

    for name, model in build_models(meta["class_weights"]).items():
        if only and name not in only:
            continue
        X_fit, y_fit = (subsample(X_train, y_train, svm_max_rows) if name == "svm"
                        else (X_train, y_train))
        print(f"\n=== {name} (training on {len(y_fit):,} rows) ===")

        start = time.time()
        model.fit(X_fit, y_fit)
        train_info = {"train_seconds": round(time.time() - start, 2), "train_rows": int(len(y_fit))}

        evaluate_model(name, model.predict, X_test, y_test, meta, artifacts_dir, extra, train_info)
        joblib.dump(model, out / "models" / f"{name}.joblib")

    compare_models(artifacts_dir)
    save_curves(artifacts_dir)
    if deploy_model:
        save_deployment_model(deploy_model, artifacts_dir)


def main():
    parser = argparse.ArgumentParser(description="Stage 2: train classical ML models")
    parser.add_argument("--artifacts", default=ARTIFACTS_DIR)
    parser.add_argument("--models", nargs="+",
                        choices=["random_forest", "logistic_regression", "svm", "gradient_boosting"],
                        help="train only these models (default: all)")
    parser.add_argument("--svm-max-rows", type=int, default=20_000,
                        help="training rows for the SVM (0 = all; slow on large data)")
    parser.add_argument("--extra-test", help="optional second test file, e.g. KDDTest+.txt")
    parser.add_argument("--deploy-model", default="gradient_boosting",
                        help="model saved with the preprocessor for the app ('' to skip)")
    parser.add_argument("--deploy-only", action="store_true",
                        help="skip training; only rebuild resources/artifacts/nids_model.joblib")
    args = parser.parse_args()
    if args.deploy_only:
        save_deployment_model(args.deploy_model, args.artifacts)
        return
    train_all(args.artifacts, args.models, args.svm_max_rows, args.extra_test, args.deploy_model)


if __name__ == "__main__":
    main()
