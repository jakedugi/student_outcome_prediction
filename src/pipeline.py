from __future__ import annotations
from typing import List, Dict, Any
import pandas as pd

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

    # run
    @utils.timer
    def run(self, semesters: int = 2) -> List[Dict[str, Any]]:
        """Train configured models and return metrics for each"""
        raw_df = self.loader.load()
        df = self.pre.fit_transform(raw_df)
        df = self.pre.semester_features(df, semesters)

        X_train, X_test, y_train, y_test = make_split(df)

        results: List[Dict[str, Any]] = []
        for key in self.settings.models:
            ModelCls = MODEL_REGISTRY[key]
            logger.info("\nTraining %s ...", key)
            model = ModelCls()
            model.fit(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)
            results.append(metrics)

        return sorted(results, key=lambda x: x["accuracy"], reverse=True)