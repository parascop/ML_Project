# ML-Based Network Intrusion Detection System

College mini project. A **logistic regression** model classifies network connections as **Normal** or
**Attack**, and an interactive Streamlit dashboard makes predictions and shows the results as graphs.
(The `ml/` code also trains four other models for comparison; they are not part of the dashboard.)

The full write-up (dataset, every pipeline step, results, limitations) is in
**[resources/docs/project_guide.md](resources/docs/project_guide.md)**.

## Project structure

| Folder | Contents |
|---|---|
| [`ml/`](ml) | **Data + machine learning**: datasets (`ml/data/`), preprocessing and feature selection, classical models, PyTorch network, evaluation |
| [`backend/`](backend) | **Prediction service**: loads the saved model and makes predictions (`predictor.py`, no UI code) |
| [`frontend/`](frontend) | **User interface**: Streamlit dashboard (`app.py`) and its styling/animations (`ui.py`) |
| [`resources/`](resources) | **Artifacts + documents**: trained models, metrics, plots (`artifacts/`) and the project guide (`docs/`) |

```
ml/  ──trains──▶  resources/artifacts/  ──loaded by──▶  backend/  ──used by──▶  frontend/
```

## Quick start

Python 3.11. Run all commands from this folder (the project root).

```bash
git clone https://github.com/parascop/ML_Project.git
cd ML_Project
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run frontend/app.py
```

The trained NSL-KDD models are included in `resources/artifacts/`, so the dashboard works straight away.

## Retraining

Download `KDDTrain+.txt` and `KDDTest+.txt` from <https://www.unb.ca/cic/datasets/nsl.html> into
`ml/data/nsl-kdd/` (CICIDS2017 instructions are in the guide), then:

```bash
python -m ml.preprocessing --dataset nslkdd --data ml/data/nsl-kdd/KDDTrain+.txt
python -m ml.classical_models --extra-test ml/data/nsl-kdd/KDDTest+.txt
python -m ml.deep_learning --extra-test ml/data/nsl-kdd/KDDTest+.txt
```

## Results (NSL-KDD, logistic regression)

The dashboard deploys a **logistic regression** model.

| Test set | Accuracy | Precision | Recall | F1 | False alarms |
|---|---|---|---|---|---|
| Held-out 20% of KDDTrain+ | 97.07% | 97.72% | 95.93% | 0.9682 | 1.95% |
| KDDTest+ (separate file, includes unseen attack types) | 74.59% | 91.33% | 61.16% | 0.7326 | 7.67% |

KDDTest+ contains 17 attack types that never appear in training, and only about a third (34.1%) of
those attacks are detected. A supervised model cannot be relied on to catch zero-day attacks.

The `ml/` code also trains Random Forest, SVM, Gradient Boosting and a PyTorch network for comparison.
They are not part of the dashboard; their results, and the other limitations, are in the
[project guide](resources/docs/project_guide.md).
