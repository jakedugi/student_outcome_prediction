from __future__ import annotations
from typing import Any, Dict
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
            self.estimator.fit(X, y)
            return self

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return self.estimator.predict(X)

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
QDA = _wrap("qda", QuadraticDiscriminantAnalysis)
NaiveBayes = _wrap("naive_bayes", GaussianNB)