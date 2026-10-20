"""
Evaluation helpers shared by every model (classical and deep learning).

  compute_metrics()   accuracy, precision, recall, F1, confusion matrix, per-class report
  print_metrics()     readable console summary
  save_results()      one JSON file per model in resources/artifacts/results/
  evaluate_model()    all of the above for one trained model (test set + optional extra test set)
  compare_models()    comparison table + chart of every saved model
  save_curves()       ROC / precision-recall curves and score histograms for the app's graphs

Compare all models trained so far:
    python -m ml.evaluation
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from ml.preprocessing import ARTIFACTS_DIR, load_extra_test_set, load_processed_data


def compute_metrics(y_true, y_pred, class_names):
    """
    Binary task: precision/recall/F1 are for the Attack class (attack = positive).
    Multiclass task: they are macro averages (every class counts equally).
    """
    labels = list(range(len(class_names)))
    binary = len(class_names) == 2
    average = "binary" if binary else "macro"

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, labels=labels, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, labels=labels, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, labels=labels, average=average, zero_division=0),
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if binary:
        tn, fp, fn, tp = cm.ravel()
        # share of normal traffic wrongly flagged as an attack (false alarms)
        metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) else None
    metrics = {k: (round(float(v), 4) if v is not None else None) for k, v in metrics.items()}
    metrics["confusion_matrix"] = cm.tolist()
    metrics["per_class"] = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0)
    return metrics


def print_metrics(name, metrics, class_names, y_true=None, y_pred=None):
    summary = {k: metrics[k] for k in ["accuracy", "precision", "recall", "f1",
                                       "false_positive_rate"] if k in metrics}
    print(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in summary.items()))
    if y_true is not None:
        print(classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                    target_names=class_names, digits=4, zero_division=0))


def plot_confusion_matrix(cm, class_names, title, path):
    cm = np.array(cm)
    plt.figure(figsize=(max(4, 0.9 * len(class_names) + 2), max(3.5, 0.8 * len(class_names) + 1.5)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def save_results(name, metrics, artifacts_dir=ARTIFACTS_DIR):
    out = Path(artifacts_dir) / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.json", "w") as f:
        json.dump(metrics, f, indent=2)


def load_extra_test(path, metadata, artifacts_dir=ARTIFACTS_DIR):
    """Load a separate test file and transform it with the saved Phase 1 preprocessor."""
    preprocessor = joblib.load(Path(artifacts_dir) / "preprocessor.joblib")
    X_raw, y, raw_labels = load_extra_test_set(path, metadata)
    X = preprocessor.transform(X_raw).astype("float32")
    print(f"Extra test set {Path(path).name}: {len(y):,} rows")
    return {"name": Path(path).name, "X": X, "y": y, "raw_labels": raw_labels}


def unseen_attack_detection_rate(y_pred, raw_labels, training_raw_labels, normal_index=0):
    """Share of attacks whose TYPE never appeared in training that were still flagged as attacks."""
    unseen = ~np.isin(raw_labels, training_raw_labels)
    if not unseen.any():
        return None
    return round(float(np.mean(y_pred[unseen] != normal_index)), 4)


def evaluate_model(name, predict, X_test, y_test, metadata, artifacts_dir=ARTIFACTS_DIR,
                   extra=None, train_info=None):
    """
    Evaluate one trained model the same way for every model type.
    `predict` is a function that takes a feature matrix and returns class indices.
    `extra` is the dict returned by load_extra_test() (or None).
    """
    plots = Path(artifacts_dir) / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    class_names = metadata["class_names"]

    y_pred = predict(X_test)
    metrics = compute_metrics(y_test, y_pred, class_names)
    metrics.update(train_info or {})
    print_metrics(name, metrics, class_names, y_test, y_pred)
    plot_confusion_matrix(metrics["confusion_matrix"], class_names,
                          f"{name} - test set", plots / f"cm_{name}.png")

    if extra:
        y_pred_extra = predict(extra["X"])
        extra_metrics = compute_metrics(extra["y"], y_pred_extra, class_names)
        extra_metrics["unseen_attack_detection_rate"] = unseen_attack_detection_rate(
            y_pred_extra, extra["raw_labels"], metadata["training_raw_labels"])
        metrics["extra_test"] = extra_metrics
        print_metrics(f"{name} on {extra['name']}", extra_metrics, class_names)
        print(f"  attacks of types never seen in training that were detected: "
              f"{extra_metrics['unseen_attack_detection_rate']}")
        plot_confusion_matrix(extra_metrics["confusion_matrix"], class_names,
                              f"{name} - {extra['name']}", plots / f"cm_{name}_extra_test.png")

    save_results(name, metrics, artifacts_dir)
    return metrics


def compare_models(artifacts_dir=ARTIFACTS_DIR):
    """Build a table of all saved model results and pick the best by test F1."""
    results_dir = Path(artifacts_dir) / "results"
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        with open(path) as f:
            m = json.load(f)
        row = {"model": path.stem}
        for k in ["accuracy", "precision", "recall", "f1", "false_positive_rate", "train_seconds"]:
            if k in m:
                row[k] = m[k]
        if "extra_test" in m:
            row["extra_test_f1"] = m["extra_test"]["f1"]
            row["unseen_attack_detection"] = m["extra_test"].get("unseen_attack_detection_rate")
        rows.append(row)
    if not rows:
        print(f"No results found in {results_dir}/")
        return None

    table = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    table.to_csv(Path(artifacts_dir) / "model_comparison.csv", index=False)
    print("\nModel comparison (test set, sorted by F1):")
    print(table.to_string(index=False))

    long = table.melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1"],
                      var_name="metric", value_name="score")
    plt.figure(figsize=(9, 4.5))
    sns.barplot(data=long, x="model", y="score", hue="metric")
    low = max(0.0, long["score"].min() - 0.05)
    plt.ylim(low, 1.0)
    plt.title("Model comparison on the test set")
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(Path(artifacts_dir) / "model_comparison.png", dpi=120)
    plt.close()

    best = table.iloc[0]["model"]
    print(f"\nBest model by test F1: {best}")
    return table


def _thin(*arrays, max_points=250):
    """Keep at most max_points evenly spaced points of a curve (enough to draw it)."""
    idx = np.unique(np.linspace(0, len(arrays[0]) - 1, max_points).astype(int))
    return [np.round(a[idx], 5).tolist() for a in arrays]


def save_curves(artifacts_dir=ARTIFACTS_DIR):
    """
    Binary task only. For every saved model, compute on the held-out test set:
      ROC curve + AUC, precision-recall curve + average precision, and (for models
      that output probabilities) a histogram of attack probabilities per true class.
    Saved to resources/artifacts/curves.json, which the Streamlit app reads.
    """
    out = Path(artifacts_dir)
    _, _, X_test, y_test, meta = load_processed_data(artifacts_dir)
    if len(meta["class_names"]) != 2:
        print("Curves are only drawn for the binary task; skipped.")
        return
    curves = {}
    for path in sorted((out / "models").iterdir()):
        if path.suffix == ".joblib":
            model = joblib.load(path)
            if hasattr(model, "predict_proba"):
                scores, is_probability = model.predict_proba(X_test)[:, 1], True
            else:  # SVC without probability estimates: use its decision score
                scores, is_probability = model.decision_function(X_test), False
        elif path.suffix == ".pt":
            import torch

            from ml.deep_learning import load_network
            with torch.no_grad():
                logits = load_network(path)(torch.from_numpy(X_test))
            scores, is_probability = torch.softmax(logits, dim=1)[:, 1].numpy(), True
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, scores)
        precision, recall, _ = precision_recall_curve(y_test, scores)
        fpr, tpr = _thin(fpr, tpr)
        precision, recall = _thin(precision, recall)
        entry = {"roc_auc": round(float(roc_auc_score(y_test, scores)), 7),
                 "average_precision": round(float(average_precision_score(y_test, scores)), 7),
                 "roc": {"fpr": fpr, "tpr": tpr},
                 "pr": {"precision": precision, "recall": recall}}
        if is_probability:
            edges = np.linspace(0, 1, 21)
            entry["histogram"] = {
                "bin_start": edges[:-1].round(2).tolist(),
                "counts": {meta["class_names"][c]: np.histogram(scores[y_test == c], bins=edges)[0].tolist()
                           for c in (0, 1)}}
        curves[path.stem] = entry
        print(f"  {path.stem}: ROC AUC {entry['roc_auc']}, average precision {entry['average_precision']}")
    with open(out / "curves.json", "w") as f:
        json.dump(curves, f)
    print(f"Saved {out / 'curves.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare all trained models")
    parser.add_argument("--artifacts", default=ARTIFACTS_DIR)
    parser.add_argument("--curves", action="store_true", help="also save ROC / PR curves for the app")
    args = parser.parse_args()
    compare_models(args.artifacts)
    if args.curves:
        save_curves(args.artifacts)
