# Student Outcome Prediction

[![CI](https://github.com/jakedugi/student_outcome_prediction/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jakedugi/student_outcome_prediction/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jakedugi/student_outcome_prediction/branch/main/graph/badge.svg)](https://codecov.io/gh/jakedugi/student_outcome_prediction)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jakedugi/student_outcome_prediction/blob/main/demo.ipynb)

Train and evaluate a state-of-the-art model for predicting student outcomes using clean, modular ML pipelines.

---

## Highlights

- **One-command pipeline** – `python main.py train` runs the full pipeline: data loading, preprocessing, training, and evaluation.
- **Modular architecture** – Swap models, scalers, or input features using clean abstractions (`Preprocessor`, `BaseClassifier`, `TrainingPipeline`).
- **Terminal-first design** – Full evaluation reports and model leaderboard printed in terminal. No notebooks required.
- **Easy model registry** – Add/remove models in a single line in `src/models/registry.py`.
- **Production-ready layout** – Separation of concerns for config, training logic, and model orchestration.

---

##  Example Output (2-Semester Accuracy Leaderboard)

```bash
 Leaderboard
random_forest      -> accuracy = 0.775
xgboost            -> accuracy = 0.764
gradient_boosting  -> accuracy = 0.762
adaboost           -> accuracy = 0.750
log_reg            -> accuracy = 0.737
qda                -> accuracy = 0.708
decision_tree      -> accuracy = 0.687
neural_net         -> accuracy = 0.676
naive_bayes        -> accuracy = 0.667
svm                -> accuracy = 0.537
knn                -> accuracy = 0.489
```
---

## Quick-start

```bash
git clone https://github.com/jakedugi/student_outcome_prediction.git
cd student_outcome_prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the Kaggle CSV and place it at:

```bash
data/dataset.csv
```

Then run:
```bash
python main.py train            # Full 2-semester baseline
python main.py train --semesters 1
python main.py train --semesters 0
```

---

## Project structure

```text
student_outcome_prediction/
├── src/                       # Core pipeline modules
│   ├── config.py              # Global constants & model settings
│   ├── data_loader.py         # CSV → DataFrame loader
│   ├── preprocess.py          # Feature scaling and label encoding
│   ├── split.py               # Train-test splitting logic
│   ├── pipeline.py            # Full training + evaluation pipeline
│   ├── utils.py               # Logging + decorators
│   └── models/                # All model implementations
│       ├── base.py            # Shared BaseClassifier interface
│       ├── registry.py        # Model lookup registry
│       ├── sklearn_wrappers.py# scikit-learn classifiers
│       └── neural_net.py      # Keras neural network wrapper
├── main.py                    # CLI entry point
├── requirements.txt           # Dependencies
├── README.md                  # Project overview
├── LICENSE
└── data/                      # (git-ignored) Place your CSV here
```

---

## Dataset

**Source:** [Kaggle – Higher Education: Predictors of Student Retention](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention/data)  
**License:** CC-BY-4.0

---

## Theory and Walkthrough

Read the full blog and literature-backed discussion:

[Tutorial Blog on Medium](https://medium.com/@Jake_2287/student-outcome-prediction-36702de0f4a3)

## Interactive Demo

Check out our [interactive demo notebook](demo.ipynb) to see the model in action:
- Train and evaluate models with different semester data
- Visualize model performance with confusion matrices
- Explore feature importance using SHAP values
- Run it directly in Google Colab with one click
