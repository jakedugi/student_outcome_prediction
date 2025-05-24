# Student Outcome Prediction

[![CI](https://github.com/jakedugi/student_outcome_prediction/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jakedugi/student_outcome_prediction/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jakedugi/student_outcome_prediction/branch/main/graph/badge.svg)](https://codecov.io/gh/jakedugi/student_outcome_prediction)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jakedugi/student_outcome_prediction/blob/main/demo.ipynb)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)

Train and evaluate a state-of-the-art model for predicting student outcomes using clean, modular ML pipelines.

## 🔍 Key Findings

Our analysis revealed several interesting insights about student retention:

1. **Simple Features Beat Complex Models**: Traditional ML models (Random Forest, XGBoost) consistently outperformed neural networks, achieving 77-78% accuracy with just basic academic metrics.

2. **Early Indicators Matter**: First-semester performance is highly predictive - we achieve 74% accuracy without any second-semester data.

3. **Feature Importance**: Behavioral metrics (attendance, participation) are more predictive than pure academic performance (grades):

![Feature Importance](reports/feature_importance.png)

> 💡 **Why This Works**: Graph-based problems with clear hierarchical relationships (like student performance metrics) often benefit more from feature engineering and tree-based models than from deep learning approaches.

---

## Highlights

- **One-command pipeline** – `python main.py train` runs the full pipeline: data loading, preprocessing, training, and evaluation.
- **Modular architecture** – Swap models, scalers, or input features using clean abstractions (`Preprocessor`, `BaseClassifier`, `TrainingPipeline`).
- **Terminal-first design** – Full evaluation reports and model leaderboard printed in terminal. No notebooks required.
- **Easy model registry** – Add/remove models in a single line in `src/models/registry.py`.
- **Production-ready layout** – Separation of concerns for config, training logic, and model orchestration.

---

## Quick Start

### Option 1: Try in Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jakedugi/student_outcome_prediction/blob/main/demo.ipynb)

Just click the badge and run all cells!

### Option 2: Run Locally

```bash
git clone https://github.com/jakedugi/student_outcome_prediction.git
cd student_outcome_prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

[Follow dataset setup instructions](#dataset-setup)

Then run:
```bash
python main.py train            # Full 2-semester baseline
python main.py train --semesters 1  # First semester only
python scripts/generate_feature_importance.py  # Create importance plot
```

### Dataset Setup

You have two options to get the dataset:

#### Option 1: Using Kaggle API (Recommended)

1. Install the Kaggle package:
```bash
pip install kaggle
```

2. Set up your Kaggle credentials:
   - Go to [Kaggle.com](https://www.kaggle.com) → Account → Create API Token
   - Download `kaggle.json`
   - Place it in `~/.kaggle/` and set permissions:
```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

3. Download the dataset:
```bash
kaggle datasets download -d thedevastator/higher-education-predictors-of-student-retention
unzip higher-education-predictors-of-student-retention.zip -d data/
rm higher-education-predictors-of-student-retention.zip
```

#### Option 2: Manual Download

1. Go to [Dataset Page](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention)
2. Click "Download"
3. Extract the ZIP file
4. Place `dataset.csv` in the `data/` directory

### Running the Model

Once the dataset is in place, run:
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

**Source:** [Higher Education: Predictors of Student Retention](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention)  
**License:** CC-BY-4.0

The dataset includes:
- Academic performance metrics
- Demographic information
- Enrollment details
- Student outcomes

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

## 🎯 Features

- Predicts student retention/dropout likelihood
- Uses multiple ML models (Random Forest, XGBoost, etc.)
- Feature importance analysis with SHAP values
- Comprehensive test coverage and CI/CD pipeline
- Interactive demo notebook

## 🚀 Quick Start

### Option 1: Run in Google Colab

1. Click the "Open in Colab" badge above
2. Follow the in-notebook instructions to set up your Kaggle credentials
3. Run all cells to see the model in action

> ⚠️ The notebook requires a Kaggle account and API credentials.  
> Don't worry - it's free and takes just a minute to set up!

### Option 2: Run Locally

1. Clone the repository:
```bash
git clone https://github.com/jakedugi/student_outcome_prediction.git
cd student_outcome_prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Kaggle credentials:
   - Create an account on [Kaggle](https://www.kaggle.com)
   - Go to Account → Create API Token
   - Place the downloaded `kaggle.json` in `~/.kaggle/`
   - Make it readable only by you: `chmod 600 ~/.kaggle/kaggle.json`

4. Run the demo notebook:
```bash
jupyter notebook demo.ipynb
```

## 📊 Dataset

This project uses the ["Higher Education Predictors of Student Retention"](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention) dataset from Kaggle. The dataset includes:

- Academic performance metrics
- Demographic information
- Enrollment details
- Student outcomes

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

> Note: The dataset used in this project has its own license terms from Kaggle.

## 📚 Learn More

- [Blog Post: Why Simple Models Win](https://medium.com/@Jake_2287/student-outcome-prediction-36702de0f4a3) - Deep dive into our findings
- [Interactive Demo](demo.ipynb) - Try different models and visualize results
- [CI/CD Pipeline](.github/workflows/ci.yml) - See our testing and deployment setup

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
