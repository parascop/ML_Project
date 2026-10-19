"""
Phase 1 - Data preprocessing and feature selection.

Steps (same order as the project synopsis):
  1. Load and inspect the dataset
  2. Clean it: invalid numbers (inf), missing labels, duplicate rows
  3. Separate features from the label and drop identifier / leakage columns
  4. Split into train and test sets BEFORE fitting anything
  5. Encode + scale all features and rank them with a Random Forest; keep the top-k
  6. Fit the final encoder/scaler on the selected features (train only)
  7. Handle class imbalance on the training set only (SMOTE or class weights)
  8. Save everything the training stage and the Streamlit app need

Feature selection (5) runs before SMOTE (7) on purpose: features are ranked on
real training rows, and SMOTE then only has to create rows for the chosen features.

Run from the project root:
    python -m src.preprocessing --dataset cicids2017 --data data/cicids2017
    python -m src.preprocessing --dataset nslkdd --data data/nsl-kdd/KDDTrain+.txt
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

RANDOM_STATE = 42

# Columns that identify a connection rather than describe its behaviour.
# A model trained on them memorises the lab setup (IPs, times), so they are removed.
# "difficulty" in NSL-KDD was computed from how many classifiers got the row right,
# so it leaks information about the label.
LEAKAGE_COLUMNS = [
    "Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp",
    "Fwd Header Length.1",  # exact copy of "Fwd Header Length" in CICIDS2017
    "difficulty",
]

# NSL-KDD files have no header row.
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]

# Attack families used for the optional multiclass task.
NSL_KDD_FAMILIES = {
    "DoS": ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2", "mailbomb",
            "processtable", "udpstorm"],
    "Probe": ["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"],
    "R2L": ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient",
            "warezmaster", "sendmail", "named", "snmpgetattack", "snmpguess", "xlock",
            "xsnoop", "worm"],
    "U2R": ["buffer_overflow", "loadmodule", "perl", "rootkit", "ps", "sqlattack", "xterm",
            "httptunnel"],
}

# Classes with fewer rows than this (after sampling) are dropped in multiclass mode,
# because they cannot be split into train/test and oversampled reliably.
MIN_CLASS_ROWS = 10


# ---------------------------------------------------------------------------
# Step 1: load and inspect
# ---------------------------------------------------------------------------

def load_cicids2017(path):
    """Load one CICIDS2017 CSV, or every CSV in a folder, into one DataFrame."""
    path = Path(path)
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {path}")

    frames = []
    for f in files:
        print(f"  reading {f.name}")
        # latin-1 never fails; the original files contain a stray byte in "Web Attack" labels
        frames.append(pd.read_csv(f, encoding="latin-1", low_memory=False))
    df = pd.concat(frames, ignore_index=True)

    df.columns = df.columns.str.strip()  # original headers have leading spaces
    df = df.rename(columns={"Label": "label"})
    df["label"] = df["label"].astype("string").str.strip()
    return df


def load_nslkdd(path):
    """Load an NSL-KDD file such as KDDTrain+.txt (comma separated, no header)."""
    df = pd.read_csv(path, header=None, names=NSL_KDD_COLUMNS)
    df["label"] = df["label"].astype("string").str.strip()
    return df


def inspect(df):
    """Print a short summary of the raw data."""
    numeric = df.select_dtypes(include="number")
    print(f"  shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  numeric columns: {numeric.shape[1]}, other columns: {df.shape[1] - numeric.shape[1]}")
    print(f"  missing values: {int(df.isna().sum().sum()):,}")
    print(f"  infinite values: {int(np.isinf(numeric.to_numpy()).sum()):,}")
    print(f"  duplicate rows: {int(df.duplicated().sum()):,}")
    print("  label counts:")
    print(df["label"].value_counts().to_string())


# ---------------------------------------------------------------------------
# Steps 2 and 3: clean, build the target, drop leakage columns
# ---------------------------------------------------------------------------

def clean(df, dataset):
    """Drop leakage columns, fix invalid numbers and remove duplicate rows."""
    before = len(df)

    dropped = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    df = df.drop(columns=dropped)
    print(f"  dropped identifier/leakage columns: {dropped or 'none'}")

    # Rows without a label cannot be used (some CICIDS2017 files end with empty rows).
    df = df.dropna(subset=["label"])

    feature_cols = [c for c in df.columns if c != "label"]
    if dataset == "cicids2017":
        # Every CICIDS2017 feature is numeric; turn any junk text into NaN.
        for c in feature_cols:
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = pd.to_numeric(df[c], errors="coerce")

    # inf/-inf (e.g. "Flow Bytes/s" when duration is 0) -> NaN, later filled by the imputer.
    numeric_cols = df[feature_cols].select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).astype("float32")

    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  rows: {before:,} -> {len(df):,} after removing empty-label and duplicate rows")
    return df


def family_of(label, dataset):
    """Map a raw attack name to a broad attack family (multiclass task)."""
    if dataset == "nslkdd":
        if label == "normal":
            return "Normal"
        for family, names in NSL_KDD_FAMILIES.items():
            if label in names:
                return family
        return "Other"

    # CICIDS2017
    if label == "BENIGN":
        return "Normal"
    if label == "DDoS":
        return "DDoS"
    if label.startswith("DoS"):
        return "DoS"
    if label == "PortScan":
        return "PortScan"
    if label in ("FTP-Patator", "SSH-Patator"):
        return "BruteForce"
    if label.startswith("Web Attack"):
        return "WebAttack"
    if label == "Bot":
        return "Bot"
    return "Other"  # Infiltration and Heartbleed have only a few dozen rows


def make_target(labels, dataset, task):
    """Return integer targets and the matching class names."""
    if task == "binary":
        normal_name = "BENIGN" if dataset == "cicids2017" else "normal"
        y = (labels != normal_name).astype(int).to_numpy()
        return y, ["Normal", "Attack"]

    families = labels.map(lambda name: family_of(name, dataset))
    class_names = ["Normal"] + sorted(set(families) - {"Normal"})
    y = families.map({name: i for i, name in enumerate(class_names)}).to_numpy()
    return y, class_names


def downsample(df, max_rows):
    """Keep a stratified random sample so the project runs on a laptop."""
    if not max_rows or len(df) <= max_rows:
        return df
    frac = max_rows / len(df)
    sample = df.groupby("label", group_keys=False).sample(frac=frac, random_state=RANDOM_STATE)
    print(f"  sampled {len(sample):,} of {len(df):,} rows (stratified by label)")
    return sample.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Steps 5-6: encoding and scaling
# ---------------------------------------------------------------------------

def build_preprocessor(numeric_cols, categorical_cols):
    """Median-impute + scale numeric columns; impute + one-hot encode categorical ones."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers = [("num", numeric, list(numeric_cols))]
    if categorical_cols:
        transformers.append(("cat", categorical, list(categorical_cols)))
    return ColumnTransformer(transformers, remainder="drop")


# ---------------------------------------------------------------------------
# Step 5: Random Forest feature ranking
# ---------------------------------------------------------------------------

def rank_features(X_train, y_train, numeric_cols, categorical_cols, max_rows=50_000):
    """
    Fit a Random Forest on the training set and return one importance score per
    ORIGINAL column. The one-hot columns of a categorical feature (e.g. service_http,
    service_ftp, ...) are summed back into one score for "service", so the selected
    features are the same columns a user types into the app.
    """
    if len(X_train) > max_rows:  # a stratified subsample is enough to rank features
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=max_rows, stratify=y_train, random_state=RANDOM_STATE)

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_encoded = preprocessor.fit_transform(X_train)

    forest = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                    n_jobs=-1, random_state=RANDOM_STATE)
    forest.fit(X_encoded, y_train)

    # Which original column does each encoded column come from?
    owners = list(numeric_cols)
    if categorical_cols:
        onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        for col, categories in zip(categorical_cols, onehot.categories_):
            owners += [col] * len(categories)

    importance = pd.Series(forest.feature_importances_, index=owners)
    return importance.groupby(level=0).sum().sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Step 7: class imbalance
# ---------------------------------------------------------------------------

def apply_smote(X, y, ratio):
    """
    Oversample every minority class up to `ratio` x the size of the largest class.
    Used on the TRAINING set only; the test set keeps the real class distribution.
    """
    classes, counts = np.unique(y, return_counts=True)
    target = int(ratio * counts.max())
    strategy = {int(c): target for c, n in zip(classes, counts) if n < target}
    if not strategy:
        print("  classes already balanced enough, SMOTE skipped")
        return X, y

    # SMOTE needs k_neighbors < number of rows in the smallest class being oversampled
    smallest = min(n for c, n in zip(classes, counts) if int(c) in strategy)
    if smallest < 2:
        print("  a class has fewer than 2 rows, SMOTE skipped")
        return X, y
    smote = SMOTE(sampling_strategy=strategy, k_neighbors=min(5, smallest - 1),
                  random_state=RANDOM_STATE)
    return smote.fit_resample(X, y)


# ---------------------------------------------------------------------------
# Plots for the presentation
# ---------------------------------------------------------------------------

def plot_feature_importance(importance, selected, path):
    top = importance.head(len(selected) + 10)  # show some unselected ones too
    colors = ["tab:blue" if f in selected else "lightgray" for f in top.index]
    plt.figure(figsize=(8, 0.35 * len(top) + 1))
    sns.barplot(x=top.values, y=top.index, hue=top.index, palette=colors, legend=False)
    plt.xlabel("Random Forest importance")
    plt.ylabel("")
    plt.title("Feature importance (blue = selected)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_class_balance(before, after, class_names, path):
    data = pd.DataFrame({
        "class": class_names * 2,
        "rows": [before.get(i, 0) for i in range(len(class_names))]
                + [after.get(i, 0) for i in range(len(class_names))],
        "stage": ["before balancing"] * len(class_names) + ["after balancing"] * len(class_names),
    })
    plt.figure(figsize=(max(6, len(class_names)), 4))
    sns.barplot(data=data, x="class", y="rows", hue="stage")
    plt.title("Training set class counts")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(dataset, data_path, task="binary", max_rows=300_000, n_features=20,
        balance="smote", smote_ratio=0.5, test_size=0.2, artifacts_dir="artifacts"):
    out = Path(artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n[1] Loading and inspecting the data")
    df = load_cicids2017(data_path) if dataset == "cicids2017" else load_nslkdd(data_path)
    inspect(df)

    print("\n[2-3] Cleaning and separating features from the label")
    df = clean(df, dataset)
    df = downsample(df, max_rows)

    y, class_names = make_target(df["label"], dataset, task)
    if task == "multiclass":  # drop classes that are too small to split
        counts = pd.Series(y).value_counts()
        keep_ids = sorted(counts[counts >= MIN_CLASS_ROWS].index)
        dropped = [class_names[i] for i in range(len(class_names)) if i not in keep_ids]
        if dropped:
            print(f"  dropping classes with < {MIN_CLASS_ROWS} rows: {dropped}")
            mask = np.isin(y, keep_ids)
            df, y = df[mask].reset_index(drop=True), y[mask]
            remap = {old: new for new, old in enumerate(keep_ids)}
            y = np.array([remap[v] for v in y])
            class_names = [class_names[i] for i in keep_ids]

    X = df.drop(columns=["label"])
    print(f"  features: {X.shape[1]}, classes: {class_names}")
    print("  class counts:", dict(zip(class_names, np.bincount(y).tolist())))

    print("\n[4] Train/test split (stratified, before fitting any preprocessing)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)
    print(f"  train: {len(X_train):,} rows, test: {len(X_test):,} rows")

    # Columns that are constant in the training set carry no information.
    constant = [c for c in X_train.columns if X_train[c].nunique(dropna=True) <= 1]
    X_train, X_test = X_train.drop(columns=constant), X_test.drop(columns=constant)
    print(f"  dropped {len(constant)} constant columns: {constant}")

    categorical_cols = list(X_train.select_dtypes(exclude="number").columns)
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]
    print(f"  numeric: {len(numeric_cols)}, categorical: {categorical_cols or 'none'}")

    print(f"\n[5] Ranking features with a Random Forest (keeping top {n_features})")
    importance = rank_features(X_train, y_train, numeric_cols, categorical_cols)
    selected = list(importance.index[:n_features])
    print(importance.head(n_features).round(4).to_string())

    print("\n[6] Encoding and scaling the selected features (fitted on train only)")
    sel_numeric = [c for c in selected if c in numeric_cols]
    sel_categorical = [c for c in selected if c in categorical_cols]
    preprocessor = build_preprocessor(sel_numeric, sel_categorical)
    X_train_t = preprocessor.fit_transform(X_train[selected]).astype("float32")
    X_test_t = preprocessor.transform(X_test[selected]).astype("float32")
    print(f"  encoded feature matrix: {X_train_t.shape[1]} columns")

    print(f"\n[7] Handling class imbalance on the training set ({balance})")
    counts_before = pd.Series(y_train).value_counts().to_dict()
    class_weights = None
    if balance == "smote" and sel_categorical:
        # SMOTE interpolates between rows, which would create invalid one-hot values
        print("  categorical features are selected, so using class weights instead of SMOTE")
        balance = "class_weight"
    if balance == "smote":
        X_train_t, y_train = apply_smote(X_train_t, y_train, smote_ratio)
    elif balance == "class_weight":
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
        print(f"  class weights: {class_weights}")
    counts_after = pd.Series(y_train).value_counts().to_dict()
    print("  train counts before:", {class_names[k]: v for k, v in sorted(counts_before.items())})
    print("  train counts after: ", {class_names[k]: v for k, v in sorted(counts_after.items())})

    print(f"\n[8] Saving artifacts to {out}/")
    joblib.dump(preprocessor, out / "preprocessor.joblib")
    np.savez(out / "processed_data.npz", X_train=X_train_t, y_train=y_train,
             X_test=X_test_t, y_test=y_test)
    importance.rename("importance").to_csv(out / "feature_importance.csv", index_label="feature")

    # A few raw test rows (selected features + true class) to try in the Streamlit app
    samples = X_test[selected].copy()
    samples["true_class"] = [class_names[i] for i in y_test]
    samples.groupby("true_class", group_keys=False).head(20).to_csv(
        out / "test_samples.csv", index=False)

    metadata = {
        "dataset": dataset,
        "task": task,
        "class_names": class_names,
        "selected_features": selected,
        "numeric_features": sel_numeric,
        "categorical_features": sel_categorical,
        "categories": {c: sorted(X_train[c].dropna().unique().tolist()) for c in sel_categorical},
        "balance_method": balance,
        "smote_ratio": smote_ratio if balance == "smote" else None,
        "class_weights": class_weights,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "train_counts_before_balancing": {class_names[k]: int(v) for k, v in counts_before.items()},
        "test_counts": {class_names[k]: int(v) for k, v in pd.Series(y_test).value_counts().items()},
        # raw attack names seen in training; used to measure detection of NEW attack types
        "training_raw_labels": sorted(df.loc[X_train.index, "label"].unique().tolist()),
        "random_state": RANDOM_STATE,
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    plot_feature_importance(importance, selected, out / "feature_importance.png")
    plot_class_balance(counts_before, counts_after, class_names, out / "class_balance.png")
    for name in ["preprocessor.joblib", "processed_data.npz", "metadata.json",
                 "feature_importance.csv", "feature_importance.png", "class_balance.png",
                 "test_samples.csv"]:
        print(f"  {out / name}")
    print("\nPhase 1 complete.")
    return metadata


def to_float32(X):
    """Phase 1 stores features as float32; the deployed pipeline casts inputs the same way."""
    return np.asarray(X, dtype="float32")


def load_processed_data(artifacts_dir="artifacts"):
    """Used by the training stage: returns X_train, y_train, X_test, y_test, metadata."""
    out = Path(artifacts_dir)
    data = np.load(out / "processed_data.npz")
    with open(out / "metadata.json") as f:
        metadata = json.load(f)
    return data["X_train"], data["y_train"], data["X_test"], data["y_test"], metadata


def load_extra_test_set(path, metadata):
    """
    Load a separately recorded test file (e.g. NSL-KDD's KDDTest+.txt, which contains
    attack types that are not in KDDTrain+). Returns the raw selected features, the
    targets and the raw attack names. Transform the features with the saved preprocessor.
    """
    dataset, class_names = metadata["dataset"], metadata["class_names"]
    df = load_cicids2017(path) if dataset == "cicids2017" else load_nslkdd(path)
    df = clean(df, dataset)
    if metadata["task"] == "binary":
        y, _ = make_target(df["label"], dataset, "binary")
    else:  # keep only rows whose family exists in the training classes
        families = df["label"].map(lambda name: family_of(name, dataset))
        keep = families.isin(class_names).to_numpy()
        df, families = df[keep], families[keep]
        y = families.map({name: i for i, name in enumerate(class_names)}).to_numpy()
    return df[metadata["selected_features"]], y, df["label"].to_numpy()


def main():
    parser = argparse.ArgumentParser(description="Phase 1: preprocessing and feature selection")
    parser.add_argument("--dataset", choices=["cicids2017", "nslkdd"], default="cicids2017")
    parser.add_argument("--data", required=True,
                        help="CICIDS2017: folder of CSVs (or one CSV). NSL-KDD: KDDTrain+.txt")
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--max-rows", type=int, default=300_000,
                        help="stratified sample size after cleaning (0 = use all rows)")
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--balance", choices=["smote", "class_weight", "none"], default="smote")
    parser.add_argument("--smote-ratio", type=float, default=0.5,
                        help="oversample minority classes up to this fraction of the majority class")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()

    run(args.dataset, args.data, task=args.task, max_rows=args.max_rows,
        n_features=args.n_features, balance=args.balance, smote_ratio=args.smote_ratio,
        test_size=args.test_size, artifacts_dir=args.artifacts)


if __name__ == "__main__":
    main()
