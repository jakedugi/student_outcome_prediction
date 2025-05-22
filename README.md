# Student Outcome Prediction
How to test and build a state-of-the-art student outcome prediction model

---

## Highlights

* **CLI** – `python main.py train` trains and ranks all models in seconds.
* **Registry** – add/remove a model in one line (`src/models/registry.py`).
* **Rich leaderboard output** – no notebooks needed to compare models.

| Semester horizon | Best accuracy (2021 kaggle dataset) |
|------------------|-------------------------------------|
| 0 (new admit)    | 0.65 ← GradientBoosting |
| 1                | 0.75 ← GradientBoosting |
| 2                | 0.77 ← XGBoost |

---

## 🚀 Quick-start

```bash
git clone https://github.com/<yourhandle>/student_outcome_prediction.git
cd student_outcome_prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

# download the CSV to data/studentkaggledataset.csv (see kaggle link below)

python main.py train           # 2-semester baseline
python main.py train --semesters 1
python main.py train --semesters 0

## Project structure
src/…           core library code
main.py         CLI entry-point
artifacts/      auto-generated plots & models
data/           (git-ignored) place your CSV here

## Dataset
Kaggle – Higher Education: Predictors of Student Retention
Dataset license: CC-BY-4.0.

##  Original write-up & references
See docs/original_blog.md for the complete literature review and discussion copied from the original README.



## [Tutorial Blog](https://medium.com/@Jake_2287/student-outcome-prediction-36702de0f4a3)
