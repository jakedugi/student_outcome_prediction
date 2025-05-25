from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from .base import BaseClassifier
from ..utils import logger
import warnings

class NeuralNet(BaseClassifier):
    """Neural network classifier using Keras.
    
    Attributes:
        model_name: Model identifier
        epochs: Number of training epochs
        batch_size: Training batch size
        _build_model: Compiled Keras model (lazy initialized)
    """
    
    model_name = "neural_net"

    def __init__(self, epochs: int = 20, batch_size: int = 32) -> None:
        """Initialize neural network.
        
        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        super().__init__()  # Important: call parent init
        self.epochs = epochs
        self.batch_size = batch_size
        self._build_model = None  # lazy

    def _create_model(self, input_dim: int, num_classes: int) -> Sequential:
        """Create and compile Keras model.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of target classes
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(16, activation="relu"),
            Dense(8, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ])
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> NeuralNet:
        """Fit the neural network.
        
        Args:
            X: Feature DataFrame
            y: Target series
            
        Returns:
            self: The fitted classifier
        """
        input_dim = X.shape[1]
        num_classes = int(y.nunique())
        self._build_model = self._create_model(input_dim, num_classes)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress TF warnings
            self._build_model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=0
            )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predicted class indices
        """
        if self._build_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
            
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability estimates.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of class probabilities
            
        Raises:
            RuntimeError: If model not fitted
        """
        if self._build_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress TF warnings
            return self._build_model.predict(X, verbose=0)