"""Scikit-learn model wrappers with standardized interfaces."""
from __future__ import annotations
from typing import Any, Dict, Type, Optional
import contextlib
import warnings
from dataclasses import dataclass
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

from .base import BaseClassifier, T
from ..utils import logger

@dataclass
class ModelConfig:
    """Configuration for sklearn model wrappers.
    
    Attributes:
        name: Model identifier
        estimator_class: The sklearn estimator class
        default_params: Default hyperparameters
    """
    name: str
    estimator_class: Type
    default_params: Dict[str, Any]

@contextlib.contextmanager
def suppress_sklearn_warnings():
    """Context manager to temporarily suppress sklearn warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='.*covariance matrix.*')
        warnings.filterwarnings('ignore', category=UserWarning)
        yield

def _wrap(config: ModelConfig) -> Type[BaseClassifier]:
    """Factory to produce concrete sklearn wrapper classes.
    
    Args:
        config: Model configuration including name, class and defaults
        
    Returns:
        New wrapper class for the specified model
    """
    class SklearnWrapper(BaseClassifier):
        model_name = config.name

        def __init__(self, **kwargs: Any):
            """Initialize with optional parameter overrides."""
            super().__init__()
            params = {**config.default_params, **kwargs}
            self.estimator = config.estimator_class(**params)

        def fit(self, X: pd.DataFrame, y: pd.Series) -> T:
            """Fit the model, suppressing common warnings."""
            with suppress_sklearn_warnings():
                self.estimator.fit(X, y)
            return self

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            """Make predictions, suppressing common warnings."""
            with suppress_sklearn_warnings():
                return self.estimator.predict(X)

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            """Get probability estimates if supported."""
            if hasattr(self.estimator, 'predict_proba'):
                with suppress_sklearn_warnings():
                    return self.estimator.predict_proba(X)
            raise NotImplementedError(
                f"{self.model_name} does not support probability estimates"
            )

    SklearnWrapper.__name__ = config.name
    return SklearnWrapper

# Model configurations
MODEL_CONFIGS = [
    ModelConfig(
        "decision_tree",
        DecisionTreeClassifier,
        {}
    ),
    ModelConfig(
        "gradient_boosting",
        GradientBoostingClassifier,
        {}
    ),
    ModelConfig(
        "random_forest",
        RandomForestClassifier,
        {"n_estimators": 100}
    ),
    ModelConfig(
        "xgboost",
        XGBClassifier,
        {
            "n_estimators": 200,
            "max_depth": 10,
            "random_state": 42
        }
    ),
    ModelConfig(
        "log_reg",
        LogisticRegression,
        {"max_iter": 10000}
    ),
    ModelConfig(
        "svm",
        SVC,
        {}
    ),
    ModelConfig(
        "knn",
        KNeighborsClassifier,
        {"n_neighbors": 15}
    ),
    ModelConfig(
        "adaboost",
        AdaBoostClassifier,
        {"n_estimators": 200}
    ),
    ModelConfig(
        "qda",
        QuadraticDiscriminantAnalysis,
        {
            "reg_param": 0.9,
            "store_covariance": True
        }
    ),
    ModelConfig(
        "naive_bayes",
        GaussianNB,
        {}
    )
]

# Create wrapper classes
DecisionTree = _wrap(MODEL_CONFIGS[0])
GradientBoosting = _wrap(MODEL_CONFIGS[1]) 
RandomForest = _wrap(MODEL_CONFIGS[2])
XGBoost = _wrap(MODEL_CONFIGS[3])
LogisticReg = _wrap(MODEL_CONFIGS[4])
SVM = _wrap(MODEL_CONFIGS[5])
KNN = _wrap(MODEL_CONFIGS[6])
AdaBoost = _wrap(MODEL_CONFIGS[7])
QDA = _wrap(MODEL_CONFIGS[8])
NaiveBayes = _wrap(MODEL_CONFIGS[9])