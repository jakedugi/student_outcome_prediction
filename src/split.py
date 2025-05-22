from __future__ import annotations
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from . import config, utils

logger = utils.logger


@utils.timer
def make_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    y = df[config.TARGET]
    X = df.drop(columns=[config.TARGET])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    logger.info("Data split (train=%s, test=%s)", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test