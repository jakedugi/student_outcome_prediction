from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# paths
ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT / "data"
DEFAULT_CSV: Path = DATA_DIR / "studentkaggledataset.csv"
ARTIFACT_DIR: Path = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

# columns 
NUMERIC_COLUMNS: Sequence[str] = [
    "Application_order",
    "Age_at_enrollment",
    # 1st semester
    "Curricular_units_1st_sem_(credited)",
    "Curricular_units_1st_sem_(enrolled)",
    "Curricular_units_1st_sem_(evaluations)",
    "Curricular_units_1st_sem_(approved)",
    "Curricular_units_1st_sem_(grade)",
    "Curricular_units_1st_sem_(without_evaluations)",
    # 2nd semester
    "Curricular_units_2nd_sem_(credited)",
    "Curricular_units_2nd_sem_(enrolled)",
    "Curricular_units_2nd_sem_(evaluations)",
    "Curricular_units_2nd_sem_(approved)",
    "Curricular_units_2nd_sem_(grade)",
    "Curricular_units_2nd_sem_(without_evaluations)",
]

TARGET: str = "Target"
TEST_SIZE: float = 0.20
RANDOM_STATE: int = 42


# dataclasses
@dataclass(frozen=True, slots=True)
class TrainSettings:
    """Hyper-parameters that might change between runs."""
    test_size: float = TEST_SIZE
    random_state: int = RANDOM_STATE
    models: tuple[str, ...] = field(
        default_factory=lambda: (
            "decision_tree",
            "gradient_boosting",
            "random_forest",
            "xgboost",
            "log_reg",
            "svm",
            "knn",
            "adaboost",
            "qda",
            "naive_bayes",
            "neural_net",
        )
    )