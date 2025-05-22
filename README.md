# Student Outcome Prediction
Train and evaluate a state-of-the-art model for predicting student outcomes using ML pipelines.

---

## Highlights

* **One-command training** – Run `python main.py train` to load, preprocess, train, and evaluate all models across semesters.
* **Modular design** – Swap models, scalers, or features with minimal edits via `Preprocessor`, `BaseClassifier`, and `TrainingPipeline`.
* **No notebooks required** – See rich evaluation metrics and model rankings directly in the terminal.
* **Fast model extension** – Add or remove models with one line in `src/models/registry.py`.
* **Production-ready layout** – Designed for clarity, reproducibility, and maintainability.

---

# Example Output

| Semester horizon | Best accuracy (2021 Kaggle dataset) |
|------------------|--------------------------------------|
| 0 (new admit)    | 0.65 ← GradientBoosting              |
| 1                | 0.75 ← GradientBoosting              |
| 2                | 0.77 ← XGBoost                       |

---

## Quick-start

```bash
git clone https://github.com/jakedugi/student_outcome_prediction.git
cd student_outcome_prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the Kaggle CSV and save it to:
```bash
data/studentkaggledataset.csv
```

Then run:
```bash
python main.py train           # 2-semester baseline
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
└── data/                      # (git-ignored) Put your Kaggle CSV here
```
---

## Dataset

Kaggle: [Higher Education: Predictors of Student Retention](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention/data)

License: CC-BY-4.0

---

##  Original write-up & references
See docs/original_blog.md for the complete literature review and discussion copied from the original README.

---

## For Theory and Walkthrough::

#### [Tutorial Blog on Medium](https://medium.com/@Jake_2287/student-outcome-prediction-36702de0f4a3)
