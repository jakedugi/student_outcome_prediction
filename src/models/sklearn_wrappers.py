from __future__ import annotations
from typing import Any, Dict
import warnings
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

from .base import BaseClassifier
from ..utils import logger


def _wrap(name: str, estimator_cls, **default_kwargs) -> type[BaseClassifier]:
    """Factory to produce tiny concrete wrapper classes in 3 lines"""

    class _Model(BaseClassifier):
        model_name = name

        def __init__(self, **kwargs: Any):
            params = {**default_kwargs, **kwargs}
            self.estimator = estimator_cls(**params)

        def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseClassifier":
            """Fit underlying estimator, silencing only warnings.warn() calls."""
            import warnings

            # Only suppress warnings.warn(); do NOT override showwarning()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.estimator.fit(X, y)

            return self

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            # Suppress warnings during prediction
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                warnings.filterwarnings('ignore', message='.*covariance matrix.*')
                return self.estimator.predict(X)

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            """Get probability estimates - required for SHAP compatibility."""
            if hasattr(self.estimator, 'predict_proba'):
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    warnings.filterwarnings('ignore', message='.*covariance matrix.*')
                    return self.estimator.predict_proba(X)
            raise NotImplementedError(f"{self.model_name} does not support probability estimates")

        # Make the estimator callable for SHAP
        def __call__(self, X: pd.DataFrame) -> np.ndarray:
            """Make the model callable for SHAP compatibility."""
            try:
                return self.predict_proba(X)
            except NotImplementedError:
                return self.predict(X)

    _Model.__name__ = name  # nice repr in docs
    return _Model


DecisionTree = _wrap("decision_tree", DecisionTreeClassifier)
GradientBoosting = _wrap("gradient_boosting", GradientBoostingClassifier)
RandomForest = _wrap("random_forest", RandomForestClassifier, n_estimators=100)
XGBoost = _wrap("xgboost", XGBClassifier, n_estimators=200, max_depth=10, random_state=42)
LogisticReg = _wrap("log_reg", LogisticRegression, max_iter=10000)
SVM = _wrap("svm", SVC)
KNN = _wrap("knn", KNeighborsClassifier, n_neighbors=15)
AdaBoost = _wrap("adaboost", AdaBoostClassifier, n_estimators=200)
QDA = _wrap("qda", QuadraticDiscriminantAnalysis, reg_param=0.9, store_covariance=True)  # Much higher regularization
NaiveBayes = _wrap("naive_bayes", GaussianNB)