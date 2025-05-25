"""Base model wrapper classes and interfaces."""
from __future__ import annotations
import abc
from typing import Dict, Any, Optional, List, Union, TypeVar
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.base import BaseEstimator
from .. import utils  # import from parent package
from .. import config

logger = utils.logger

# Type variable for the classifier itself
T = TypeVar('T', bound='BaseClassifier')

class BaseClassifier(abc.ABC):
    """Abstract base class for all model wrappers.
    
    This class defines the standard interface that all model wrappers must implement.
    It provides common functionality for model evaluation, prediction storage,
    and metric calculation.
    
    Attributes:
        model_name (str): Name identifier for the model
        _last_predictions (Optional[np.ndarray]): Cache of last predictions made
        estimator (BaseEstimator): The underlying sklearn estimator
    """

    model_name: str = "base"

    def __init__(self) -> None:
        """Initialize the model wrapper."""
        self._last_predictions: Optional[np.ndarray] = None
        self.estimator: Optional[BaseEstimator] = None

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> T:
        """Fit the model to training data.
        
        Args:
            X: Feature DataFrame
            y: Target series
            
        Returns:
            self: The fitted classifier
        """
        ...

    @abc.abstractmethod 
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Feature DataFrame to predict on
            
        Returns:
            Array of predictions
        """
        ...

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance on test data.
        
        Calculates standard classification metrics and stores predictions
        for later analysis.
        
        Args:
            X: Feature DataFrame
            y_true: True target values
            
        Returns:
            Dictionary containing:
                - accuracy: Overall accuracy score
                - confusion_matrix: Confusion matrix
                - y_pred: Model predictions
                - model: Model name
                - model_obj: Reference to this model instance
        """
        # Get and store predictions
        y_pred = self.predict(X)
        self._last_predictions = y_pred

        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        # Log results
        logger.info("%s accuracy = %.3f", self.model_name, acc)
        logger.info(
            "\n%s",
            classification_report(
                y_true, 
                y_pred,
                target_names=self._target_names() or None,
                zero_division=1
            ),
        )
        
        return {
            "model": self.model_name,
            "accuracy": acc,
            "y_pred": y_pred,
            "confusion_matrix": cm,
            "model_obj": self
        }

    def _target_names(self) -> Optional[List[str]]:
        """Get descriptive names for target classes.
        
        Returns:
            List of class names if available, None otherwise
        """
        return None

    def get_last_predictions(self) -> np.ndarray:
        """Retrieve the most recent predictions made by this model.
        
        Returns:
            Array of predictions from last predict() or evaluate() call
            
        Raises:
            ValueError: If no predictions have been made yet
        """
        if self._last_predictions is None:
            raise ValueError(
                f"No predictions available for {self.model_name}. "
                "Call predict() or evaluate() first."
            )
        return self._last_predictions

    @abc.abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability estimates for predictions.
        
        Required for SHAP compatibility.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of class probabilities
            
        Raises:
            NotImplementedError: If model doesn't support probability estimates
        """
        ...

    def __call__(self, X: pd.DataFrame) -> np.ndarray:
        """Make model callable for SHAP compatibility.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Probability estimates if available, otherwise predictions
        """
        try:
            return self.predict_proba(X)
        except NotImplementedError:
            return self.predict(X)