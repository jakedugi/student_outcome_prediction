from __future__ import annotations
import abc
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from .. import utils  # import from parent package
from .. import config

logger = utils.logger


class BaseClassifier(abc.ABC):
    """Abstract wrapper that enforces fit / predict / evaluate signature"""

    model_name: str = "base"

    def __init__(self):
        self._last_predictions = None  # Store last predictions

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseClassifier": ...

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    # evaluate
    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, Any]:
        # Get and store predictions
        y_pred = self.predict(X)
        self._last_predictions = y_pred  # Store for later access

        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        logger.info("%s accuracy = %.3f", self.model_name, acc)
        logger.info(
            "\n%s",
            classification_report(y_true, y_pred, target_names=self._target_names() or None, zero_division=1),
        )
        
        # Return comprehensive metrics
        return {
            "model": self.model_name,
            "accuracy": acc,
            "y_pred": y_pred,  # Always include predictions
            "confusion_matrix": cm,
            "model_obj": self  # Include model object for later use
        }

    # convenience
    def _target_names(self):
        return None  # override in subclasses if LabelEncoder available
        
    def get_last_predictions(self) -> np.ndarray:
        """Get the last predictions made by this model."""
        if self._last_predictions is None:
            raise ValueError(f"No predictions available for {self.model_name}. Call predict() or evaluate() first.")
        return self._last_predictions