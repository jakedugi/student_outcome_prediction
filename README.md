# Student Outcome Prediction

[![CI](https://github.com/jakedugi/student_outcome_prediction/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jakedugi/student_outcome_prediction/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jakedugi/student_outcome_prediction/branch/main/graph/badge.svg)](https://codecov.io/gh/jakedugi/student_outcome_prediction)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jakedugi/student_outcome_prediction/blob/main/demo.ipynb)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)

---

A modular, production-grade ML pipeline that predicts student retention outcomes using academic and behavioral features. Designed to be testable, interpretable, and easy to deploy.

## Key Insights

**1. Simple Models Win**  
Tree-based models (Random Forest, XGBoost) consistently outperform neural networks on this task, reaching 77–78% accuracy.

**2. Early Data Is Powerful**  
Using only first-semester data achieves ~74% accuracy — highlighting the value of early intervention.

**3. Behavior Over Grades**  
Attendance and participation are more predictive than GPA.

![Feature Importance](reports/feature_importance.png)

[Read the full technical breakdown on Medium →](https://medium.com/@jakedugi/student-outcome-prediction-36702de0f4a3)

---

## Core Features

- **One-command CLI**: `python main.py train` triggers the full pipeline  
- **Clean abstractions**: Easily swap models, scalers, and features  
- **Fully modular**: Designed for extensibility and rapid experimentation  
- **CI/CD ready**: GitHub Actions with test coverage and linting  
- **Colab demo notebook**: Interactive and easy to run, no setup required  

---

## Quick Start

### Option 1: Run in Google Colab (Fastest)

- Open the [demo notebook](https://colab.research.google.com/github/jakedugi/student_outcome_prediction/blob/main/demo.ipynb)  
- Follow instructions to upload your Kaggle API token  
- Just click the badge and run all cells!

### Option 2: Run Locally

```bash
git clone https://github.com/jakedugi/student_outcome_prediction.git
cd student_outcome_prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Set up the Dataset

#### Option 1 – Using Kaggle CLI

Set up your Kaggle credentials:
   - Go to [Kaggle.com](https://www.kaggle.com) → Account → Create API Token
   - Download `kaggle.json`
   
```bash
pip install kaggle

mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d thedevastator/higher-education-predictors-of-student-retention
unzip higher-education-predictors-of-student-retention.zip -d data/
```

#### Option 2 – manual:

1. Go to [Dataset Page](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention)
2. Click "Download"
3. Extract the ZIP file
4. Place `dataset.csv` in the `data/` directory


#### Train the model:
```bash
python main.py train                  # Full 2-semester pipeline
python main.py train --semesters 1   # 1-semester baseline
python main.py train --semesters 0   # Entry data only
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

**Source:** [Higher Education: Predictors of Student Retention](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention)  
**License:** CC-BY-4.0

Features:
	•	Academic metrics (grades, GPA)
	•	Behavioral data (attendance, engagement)
	•	Student demographics
	•	Retention outcomes

---

## Learn More

- [Blog Post: Why Simple Models Win](https://medium.com/@Jake_2287/student-outcome-prediction-36702de0f4a3) - Deep dive into our findings
- [Interactive Demo](demo.ipynb) - Try different models and visualize results
- [CI/CD Pipeline](.github/workflows/ci.yml) - See our testing and deployment setup


