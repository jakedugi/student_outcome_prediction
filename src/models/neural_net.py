from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from .base import BaseClassifier
from ..utils import logger

class NeuralNet(BaseClassifier):
    model_name = "neural_net"

    def __init__(self, epochs: int = 20, batch_size: int = 32) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self._build_model = None  # lazy

    def _create_model(self, input_dim: int, num_classes: int):
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

    # fit
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NeuralNet":
        input_dim = X.shape[1]
        num_classes = int(y.nunique())
        self._build_model = self._create_model(input_dim, num_classes)
        self._build_model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self._build_model.predict(X, verbose=0)
        return proba.argmax(axis=1)