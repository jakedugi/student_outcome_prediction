# Student Outcome Prediction

Train and evaluate a state-of-the-art model for predicting student outcomes using clean, modular ML pipelines.

---

## 🚀 Highlights

- **One-command pipeline** – `python main.py train` runs the full pipeline: data loading, preprocessing, training, and evaluation.
- **Modular architecture** – Swap models, scalers, or input features using clean abstractions (`Preprocessor`, `BaseClassifier`, `TrainingPipeline`).
- **Terminal-first design** – Full evaluation reports and model leaderboard printed in terminal. No notebooks required.
- **Easy model registry** – Add/remove models in a single line in `src/models/registry.py`.
- **Production-ready layout** – Separation of concerns for config, training logic, and model orchestration.

---

## 🧪 Example Output (Accuracy by Semester)

| Semester Horizon | Best Accuracy (Kaggle Dataset) | Top Model           |
|------------------|-------------------------------|---------------------|
| 0 (New Admit)    | 0.65                          | GradientBoosting    |
| 1 Semester       | 0.74                          | XGBoost             |
| 2 Semesters      | 0.78                          | RandomForest        |

---

## ⚡ Quick Start

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
python main.py train             # Full 2-semester baseline
python main.py train --semesters 1
python main.py train --semesters 0
```

---

## 🗂 Project Structure

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

## 📊 Dataset

**Source:** [Kaggle – Higher Education: Predictors of Student Retention](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention/data)  
**License:** CC-BY-4.0

---

## 📚 Theory & Walkthrough

Read the full blog and literature-backed discussion:

👉 [Tutorial Blog on Medium](https://medium.com/@Jake_2287/student-outcome-prediction-36702de0f4a3)
