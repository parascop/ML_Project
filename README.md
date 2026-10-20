# ML-Based Network Intrusion Detection System

College mini project. Machine-learning models classify network connections as **Normal** or
**Attack**. Four Scikit-learn models are compared with a PyTorch neural network, and an interactive
Streamlit dashboard makes predictions and shows the results as graphs.

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

## Results (NSL-KDD, held-out 20% test set)

| Model | Accuracy | F1 | False-positive rate |
|---|---|---|---|
| Gradient Boosting (deployed) | 0.9993 | 0.9992 | 0.04% |
| Random Forest | 0.9988 | 0.9988 | 0.06% |
| Neural network (PyTorch) | 0.9957 | 0.9953 | 0.31% |
| SVM | 0.9823 | 0.9808 | 0.74% |
| Logistic Regression | 0.9707 | 0.9682 | 1.95% |

On the separate KDDTest+ file, which contains 17 attack types never seen in training, F1 drops to
0.73–0.78. Only about a third of the attacks of those unseen types are detected. A supervised model
cannot be relied on to catch zero-day attacks. Details and other limitations are in the
[project guide](resources/docs/project_guide.md).
