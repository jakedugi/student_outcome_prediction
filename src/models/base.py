from __future__ import annotations
import abc
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from . import utils  # relative import inside models package
from .. import config

logger = utils.logger


class BaseClassifier(abc.ABC):
    """Abstract wrapper that enforces fit / predict / evaluate signature"""

    model_name: str = "base"

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseClassifier": ...

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    # evaluate
    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, Any]:
        y_pred = self.predict(X)
        acc = accuracy_score(y_true, y_pred)
        logger.info("%s accuracy = %.3f", self.model_name, acc)
        logger.info(
            "\n%s",
            classification_report(y_true, y_pred, target_names=self._target_names() or None, zero_division=1),
        )
        return {"model": self.model_name, "accuracy": acc}

    # convenience
    def _target_names(self):
        return None  # override in subclasses if LabelEncoder available