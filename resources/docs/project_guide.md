# Project guide: ML-Based Network Intrusion Detection System

Classifies network flows as **Normal** or **Attack** with machine learning. Classical
Scikit-learn models are compared with a small PyTorch neural network, and an interactive
Streamlit dashboard makes predictions and shows how the models perform.

| Stage | What | Status |
|---|---|---|
| 1 | Preprocessing + feature selection (`ml/preprocessing.py`) | done |
| 2 | Classical models + evaluation (`ml/classical_models.py`, `ml/evaluation.py`) | done (NSL-KDD) |
| 3 | PyTorch feed-forward network (`ml/deep_learning.py`) | done (NSL-KDD) |
| 4 | Streamlit dashboard (`frontend/app.py`, `frontend/ui.py`) | done |

## Project structure

```
ML_Project/
├── ml/                        data + machine learning
│   ├── data/                  datasets (not in git): cicids2017/, nsl-kdd/, sample/ (fake test files)
│   ├── preprocessing.py       Phase 1: cleaning, split, encoding, scaling, SMOTE, feature selection
│   ├── classical_models.py    Stage 2: Random Forest, Logistic Regression, SVM, Gradient Boosting
│   ├── deep_learning.py       Stage 3: PyTorch feed-forward network
│   ├── evaluation.py          metrics, confusion matrices, model comparison, ROC/PR curves
│   └── make_sample_data.py    fake data, only for checking that the code runs
├── backend/
│   └── predictor.py           loads the saved model, makes predictions (no UI code)
├── frontend/
│   ├── app.py                 Stage 4: Streamlit dashboard
│   └── ui.py                  styling: animated threat dial, header, event feed
├── resources/
│   ├── resources/artifacts/             saved models, metrics, plots, curves (written by ml/, read by backend/)
│   └── docs/                  this guide
├── .streamlit/config.toml     dark theme (must stay in the project root)
├── requirements.txt
└── README.md
```

Run every command from the project root (the folder that contains `ml/`, `backend/` and `frontend/`).

## Installation

You need Python 3.11.

```bash
git clone https://github.com/parascop/ML_Project.git
cd ML_Project
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The repository already contains the trained models and results in `resources/artifacts/`, so after installing
you can run `streamlit run frontend/app.py` straight away. The steps below are needed only to retrain.

## Getting the dataset

### CICIDS2017 (main dataset)

1. Open the Canadian Institute for Cybersecurity page: <https://www.unb.ca/cic/datasets/ids-2017.html>
2. Fill in the short form and download **`MachineLearningCSV.zip`** (a few hundred MB zipped,
   roughly 1 GB extracted). The bigger "GeneratedLabelledFlows" version also works; its
   IP/timestamp columns are removed automatically.
3. Extract it and copy the 8 CSV files (e.g. `Monday-WorkingHours.pcap_ISCX.csv`,
   `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`, ...) into `ml/data/cicids2017/`.

The loader reads every `*.csv` in that folder and joins them. You can also point it at one
file to start smaller, e.g. only the Friday DDoS file.

### NSL-KDD (optional, much smaller)

Download `KDDTrain+.txt` (and `KDDTest+.txt` if you want it later) from
<https://www.unb.ca/cic/datasets/nsl.html> into `ml/data/nsl-kdd/`.
These files have no header row; the code adds the 41 column names.

## Phase 1: preprocessing and feature selection

Run from inside `nids-mini/` with the virtual environment active:

```bash
# CICIDS2017, binary (normal vs attack) - the main setting
python -m ml.preprocessing --dataset cicids2017 --data ml/data/cicids2017

# NSL-KDD instead
python -m ml.preprocessing --dataset nslkdd --data ml/data/nsl-kdd/KDDTrain+.txt

# Multiclass (attack families) if time permits
python -m ml.preprocessing --dataset cicids2017 --data ml/data/cicids2017 --task multiclass
```

Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--max-rows` | 300000 | stratified sample size after cleaning, so it fits on a laptop (`0` = all rows) |
| `--n-features` | 20 | how many features the Random Forest keeps |
| `--balance` | `smote` | `smote`, `class_weight` or `none` |
| `--smote-ratio` | 0.5 | SMOTE raises each minority class to this fraction of the majority class |
| `--test-size` | 0.2 | share of rows held out for testing |

Running Phase 1 again overwrites `resources/artifacts/`.

### What each step does (for the presentation)

1. **Load and inspect.** Print rows, columns, missing values, infinite values, duplicates and label counts.
2. **Clean.** Turn `inf`/`-inf` (e.g. `Flow Bytes/s` when the duration is 0) and junk text into
   missing values, drop rows without a label, remove exact duplicate rows.
3. **Separate features and label; remove leakage.** Drop identifiers (`Flow ID`, IPs, source port,
   `Timestamp`), the duplicated `Fwd Header Length.1` column and NSL-KDD's `difficulty` (it was
   computed from the label). Build the target: `0 = Normal`, `1 = Attack`, or attack families in multiclass mode.
4. **Split first.** Stratified 80/20 train/test split *before* anything is fitted, so the test set
   never influences the imputer, scaler, feature selection or SMOTE. Columns that are constant in the
   training set are removed.
5. **Feature selection.** Encode + scale every feature, train a Random Forest on the training set and
   rank features by importance. One-hot columns of a categorical feature are added back together
   (e.g. all `service_*` columns count as `service`), so the selected features are real input columns
   the app can ask for. The top `--n-features` are kept.
6. **Encode and scale the selected features.** Missing numbers get the training median, numbers are
   standardised, categorical columns are one-hot encoded (unknown categories are ignored). This
   `ColumnTransformer` is fitted on training data only and saved, so the app transforms input exactly the same way.
7. **Class imbalance, on the training set only.** SMOTE creates synthetic minority rows until each
   minority class reaches `--smote-ratio` × the majority class. The test set keeps the real class mix.
   If a categorical feature was selected (NSL-KDD), SMOTE would produce invalid in-between one-hot
   values, so balanced class weights are used instead and saved for the models.
8. **Save artifacts** (below).

### Phase 1 outputs (`resources/artifacts/`)

| File | Contents |
|---|---|
| `preprocessor.joblib` | fitted imputer/scaler/encoder for the selected features |
| `processed_data.npz` | `X_train`, `y_train` (after balancing), `X_test`, `y_test` (transformed) |
| `metadata.json` | class names, selected features, categories, balancing method, class weights, row counts |
| `feature_importance.csv` / `.png` | Random Forest ranking of all features (blue = selected) |
| `class_balance.png` | training class counts before and after balancing |
| `test_samples.csv` | a few raw test rows per class, to try in the Streamlit app later |

Later stages load these with `ml.preprocessing.load_processed_data()`, so every model uses the
same split and labels.

### Testing the code without the real dataset

```bash
python ml/make_sample_data.py      # writes FAKE files to ml/data/sample/
python -m ml.preprocessing --dataset cicids2017 --data ml/data/sample/cicids2017_sample.csv
python -m ml.preprocessing --dataset nslkdd --data ml/data/sample/nslkdd_sample.txt
```

The fake files copy the real column layouts (leading spaces in headers, `inf` values, duplicate
rows, constant columns) but the values are random numbers. They only show that the code runs;
**do not report any number obtained from them.** Run Phase 1 on the real data afterwards, since it
overwrites `resources/artifacts/`.

## Stage 2: classical models

```bash
python -m ml.classical_models                                         # CICIDS2017 or NSL-KDD
python -m ml.classical_models --extra-test ml/data/nsl-kdd/KDDTest+.txt   # NSL-KDD: also test on KDDTest+
python -m ml.evaluation --curves                                      # rebuild comparison table + ROC/PR curves
```

* All models load the **same** `processed_data.npz` from Phase 1 (same split, same labels).
* If Phase 1 chose class weights, every model gets them (`class_weight=...`); after SMOTE they are not needed.
* **SVM** uses an RBF kernel. Its training time grows roughly with the square of the rows, so it is
  trained on a stratified 20,000-row subsample of the training set (`--svm-max-rows`). It is still
  tested on the full test set.
* **Gradient Boosting** is Scikit-learn's `HistGradientBoostingClassifier`, the fast histogram-based
  version of gradient boosting.
* Default hyperparameters are used; no tuning is done on the test set.

Metrics (binary task): precision, recall and F1 are for the **Attack** class. The false-positive rate
is the share of normal traffic flagged as an attack (false alarms). For multiclass they are macro
averages, and the per-class report is printed and saved.

Outputs: `resources/artifacts/models/<model>.joblib`, `resources/artifacts/results/<model>.json` (all metrics, confusion
matrix, per-class report), `resources/artifacts/plots/cm_<model>.png`, `resources/artifacts/model_comparison.csv/.png`, and
`resources/artifacts/curves.json` (ROC and precision-recall curves, AUC, score histograms of every model on the
test set; used by the app's charts).

## Stage 3: neural network (PyTorch)

```bash
python -m ml.deep_learning                                          # CICIDS2017 or NSL-KDD
python -m ml.deep_learning --extra-test ml/data/nsl-kdd/KDDTest+.txt    # NSL-KDD: also test on KDDTest+
```

* Same `processed_data.npz` as the classical models, evaluated by the same `evaluate_model()`
  function, and added to the same comparison table.
* Architecture: `input -> 64 -> ReLU -> Dropout(0.2) -> 32 -> ReLU -> Dropout(0.2) -> one output per class`.
  Cross-entropy loss (with the Phase 1 class weights when present), Adam optimiser (learning rate 0.001),
  batch size 256.
* **Early stopping:** 10% of the *training* set is held out as a validation set. Training stops when
  the validation loss has not improved for 5 epochs (max 100), and the best weights are kept. The test
  set is never seen during training.
* Seeds are fixed, so the run is repeatable on the same machine.

Outputs: `resources/artifacts/models/neural_network.pt` (weights + layer sizes; reload with
`ml.deep_learning.load_network()`), `resources/artifacts/plots/nn_training_curve.png`, plus the same results
JSON (including the loss/accuracy history per epoch) and confusion matrices as the other models.

## Stage 4: Streamlit app

### Saving the model with its preprocessing

At the end of `python -m ml.classical_models`, the chosen model is joined with the fitted Phase 1
preprocessor into **one** Scikit-learn `Pipeline` and saved as `resources/artifacts/nids_model.joblib`:

```
raw selected features -> preprocessor (impute, scale, one-hot) -> cast to float32 -> Gradient Boosting
```

The frontend never touches the model directly: it calls `backend/predictor.py`, which loads this
pipeline, so typed or uploaded values are transformed exactly as the training data was. The backend has
no Streamlit code, so it can also be used from a script or notebook. (Checked: on all 25,195 raw test rows the pipeline gives the same predictions as
the evaluated model on the preprocessed test set.) The file also stores the selected feature names,
the categories for the drop-downs, the class names and the test metrics shown in the sidebar.

```bash
python -m ml.classical_models --deploy-only                          # rebuild it without retraining
python -m ml.classical_models --deploy-only --deploy-model random_forest   # deploy another model
```

### Running the app

```bash
streamlit run frontend/app.py
```

It opens at <http://localhost:8501> with a dark "security console" theme (`.streamlit/config.toml`).
The sidebar shows the deployed model, its test scores and the zero-day limitation. Five tabs:

| Tab | What you can do |
|---|---|
| 🎯 **Analyse** | Pick a held-out test connection (or 🎲 *Random*, filtered by Normal/Attack) and press *Analyse*. A radar scan runs, the speed dial does a full calibration sweep and settles on the model's attack probability, and the verdict appears. The true label is revealed only afterwards. A chart shows how far each feature is from the training average. |
| 🎛️ **Simulator** | Sliders, drop-downs and number fields for all 20 features, with presets (training median, typical normal, typical attack). The live dial moves up or down from its last value whenever you change something. A **what-if curve** sweeps one feature and plots how the attack probability changes. |
| 📡 **Live replay** | Streams randomly ordered test connections to the model one by one: live counters, a probability timeline and an event feed. This replays recorded features; it is not packet capture. |
| 📂 **Batch upload** | Upload a CSV (or press *Use demo file*): counts, a prediction donut chart, a histogram of attack probabilities, a table with probability bars, and a CSV download of the predictions. |
| 📊 **Model insights** | Graphs of the ML results, in sub-tabs: model comparison and KDDTest+ results, confusion matrix · ROC and precision-recall curves for all models (zoom/pan, AUC table) and score distributions · feature importance with cumulative importance · neural-network loss and accuracy per epoch, training times · dataset class balance. |

All graphs are drawn with Altair (installed with Streamlit) from files in `resources/artifacts/`; the dial is
HTML/SVG/JavaScript in `frontend/ui.py`. Nothing in the interface changes the predictions.

Demo idea: in the *Simulator*, press *Typical normal* (the dial drops to about 0%), then *Typical
attack* (it swings up to about 100%). Then pick `serror_rate` in the what-if curve to show where the
model's decision changes.

## Results (NSL-KDD, binary)

Phase 1 settings: `KDDTrain+.txt`, stratified 80/20 split (100,778 train / 25,195 test rows), top 20
features (17 numeric + `protocol_type`, `service`, `flag`), class weights (SMOTE skipped because
categorical features were selected). Produced by the commands above on this machine:

**Held-out 20% of KDDTrain+**

| Model | Accuracy | Precision | Recall | F1 | False-positive rate | Train time |
|---|---|---|---|---|---|---|
| Gradient Boosting | 0.9993 | 0.9996 | 0.9989 | 0.9992 | 0.0004 | 2.8 s |
| Random Forest | 0.9988 | 0.9993 | 0.9982 | 0.9988 | 0.0006 | 0.9 s |
| Neural network (PyTorch) | 0.9957 | 0.9964 | 0.9943 | 0.9953 | 0.0031 | 9.7 s (42 epochs) |
| SVM (20k-row subsample) | 0.9823 | 0.9913 | 0.9706 | 0.9808 | 0.0074 | 0.7 s |
| Logistic Regression | 0.9707 | 0.9772 | 0.9593 | 0.9682 | 0.0195 | 0.3 s |

**KDDTest+ (separate test file, includes attack types absent from training)**

| Model | Accuracy | Precision | Recall | F1 | False-positive rate | Unseen attack types detected |
|---|---|---|---|---|---|---|
| Neural network (PyTorch) | 0.7889 | 0.9512 | 0.6631 | 0.7815 | 0.0450 | 35.4% |
| Gradient Boosting | 0.7850 | 0.9689 | 0.6430 | 0.7730 | 0.0273 | 30.8% |
| Random Forest | 0.7696 | 0.9678 | 0.6157 | 0.7526 | 0.0271 | 29.0% |
| SVM | 0.7675 | 0.9596 | 0.6176 | 0.7515 | 0.0344 | 35.2% |
| Logistic Regression | 0.7459 | 0.9133 | 0.6116 | 0.7326 | 0.0767 | 34.1% |

**Best model on the held-out split: Gradient Boosting** (F1 0.9992, fewest false alarms), with Random
Forest very close behind. The neural network comes third, ahead of the SVM and Logistic Regression.

On KDDTest+ the order changes: the **neural network** has the highest F1 (0.7815) and recall, while
Gradient Boosting has the higher precision and fewer false alarms (2.7% vs 4.5%). No model is best on
every metric, so the choice depends on what matters more: catching more attacks or raising fewer false
alarms. For the demo app, Gradient Boosting is the default: it wins on the main test set, has the lowest
false-positive rate there (and about 2.7% on KDDTest+, level with Random Forest), and is easy to save
together with its preprocessing.

What the two tables show:

* On a random split of the same file the task is nearly solved (F1 ≈ 0.999). The test rows come from
  the same distribution and the same attack types as the training rows.
* On KDDTest+ every model drops to F1 ≈ 0.73-0.78. Precision stays high but recall falls to about
  0.61-0.66, so many attacks are labelled normal.
* 3,750 of the 12,833 attacks in KDDTest+ (29%) belong to 17 attack types that never occur in
  KDDTrain+ (e.g. `mscan`, `apache2`, `processtable`, `snmpguess`, `saint`). Every model, including
  the neural network, detects only about 29-35% of them. This is the zero-day limitation measured,
  not just stated.
* Model selection uses the same test set that is reported, so the "best model" score is slightly
  optimistic. The differences between the top three models are small (under 0.5 percentage points of F1).
* The deep-learning model does not beat the tree models on the held-out split (a common result on
  tabular data like this). On KDDTest+ it is slightly ahead on F1, but the margin (0.8 points over
  Gradient Boosting) is small for a single run.

These numbers come from one run with `random_state=42`. Results will be different on CICIDS2017.

## Limitations

* **Unseen (zero-day) attacks.** All models here are supervised: they learn the attack types in the
  training data. A new attack whose traffic looks unlike those types can be classified as normal.
  On NSL-KDD only about a third of the attacks of unseen types in KDDTest+ were detected (see Results).
  This project does not claim to detect zero-day attacks.
* **NSL-KDD is old.** It is derived from 1998 DARPA traffic. Good scores on it do not mean the
  model would work on a modern network.
* **Optimistic scores.** CICIDS2017 was recorded in a lab over five days. A random train/test split
  puts very similar flows from the same attack session in both sets, so test scores are higher than
  what a real network would give. Published analyses have also found labelling errors in the
  original CICIDS2017 files.
* **Sampling.** By default 300,000 rows are used for speed. Rare classes (Infiltration, Heartbleed,
  SQL injection) have only a handful of rows; in multiclass mode they are grouped into larger families
  or dropped if fewer than 10 rows remain.
* **SMOTE rows are synthetic.** They are interpolations of real minority rows, not new attacks.
* **No live capture.** The app classifies flow features that are typed in or taken from the dataset;
  it does not sniff packets.
