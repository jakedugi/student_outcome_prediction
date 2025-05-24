from __future__ import annotations
from typing import List, Dict, Any
import pandas as pd
import os
import warnings

from . import config, utils
from .data_loader import DataLoader
from .preprocess import Preprocessor
from .split import make_split
from .models.registry import MODEL_REGISTRY

logger = utils.logger


class TrainingPipeline:
    """
    End-to-end orchestrator:
      1. load data
      2. preprocess
      3. split
      4. train & evaluate a list of models
    """

    def __init__(self, settings: config.TrainSettings = config.TrainSettings()):
        self.settings = settings
        self.loader = DataLoader()
        self.pre = Preprocessor()
        self._last_results = None

    # run
    @utils.timer
    def run(self, semesters: int = 2) -> List[Dict[str, Any]]:
        """Train configured models and return metrics for each"""
        # Ensure data directory exists
        os.makedirs(config.DATA_DIR, exist_ok=True)
        
        # Suppress warnings during pipeline run
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            warnings.filterwarnings('ignore', message='.*covariance matrix.*')
            
            raw_df = self.loader.load()
            df = self.pre.fit_transform(raw_df)
            df = self.pre.semester_features(df, semesters)

            X_train, X_test, y_train, y_test = make_split(df)
            feature_names = X_train.columns.tolist()

            results: List[Dict[str, Any]] = []
            for key in self.settings.models:
                ModelCls = MODEL_REGISTRY[key]
                logger.info("\nTraining %s ...", key)
                model = ModelCls()
                
                # Train model
                model.fit(X_train, y_train)
                
                # Get predictions and evaluate
                y_pred = model.predict(X_test)
                metrics = model.evaluate(X_test, y_test)
                
                # Add additional data for visualization
                metrics.update({
                    'X_test': X_test,
                    'y_true': y_test,
                    'y_pred': y_pred,  # Explicitly include predictions
                    'feature_names': feature_names
                })
                results.append(metrics)

            # Store results for later use
            self._last_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)
            return self._last_results
            
    def get_best_model_predictions(self) -> Dict[str, Any]:
        """Get predictions and metrics for the best performing model."""
        if self._last_results is None:
            raise ValueError("No results available. Run the pipeline first.")
        return self._last_results[0]