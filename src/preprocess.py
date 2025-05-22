from __future__ import annotations
from typing import Dict, Any
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from dataclasses import dataclass, field
from . import config, utils

logger = utils.logger


@dataclass
class Preprocessor:
    """
    Fits scalers/encoders on training data and transforms splits consistently
    """

    standard_scaler: StandardScaler = field(
        default_factory=lambda: StandardScaler()
    )
    minmax_scaler: MinMaxScaler = field(default_factory=lambda: MinMaxScaler())
    target_encoder: LabelEncoder = field(default_factory=lambda: LabelEncoder())
    mappings_: Dict[str, Any] = field(init=False, default_factory=dict)

    # fit
    @utils.timer
    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        numeric = df[config.NUMERIC_COLUMNS]

        self.standard_scaler.fit(numeric)
        self.minmax_scaler.fit(numeric)

        # preserve mapping for README demo & inverse_transform
        self.mappings_["standard"] = dict(
            zip(config.NUMERIC_COLUMNS, zip(self.standard_scaler.mean_, self.standard_scaler.scale_))
        )
        self.mappings_["minmax"] = dict(
            zip(config.NUMERIC_COLUMNS, zip(self.minmax_scaler.data_min_, self.minmax_scaler.data_max_))
        )

        self.target_encoder.fit(df[config.TARGET])
        logger.info("Preprocessor fitted (%d numeric columns)", len(config.NUMERIC_COLUMNS))
        return self

    # transform
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies fitted transforms; returns a copy (does not modify in place)."""
        out = df.copy()
        num_scaled = self.standard_scaler.transform(out[config.NUMERIC_COLUMNS])
        out.loc[:, config.NUMERIC_COLUMNS] = self.minmax_scaler.transform(pd.DataFrame(num_scaled, columns=config.NUMERIC_COLUMNS))
        # encode target if present
        if config.TARGET in out.columns:
            out[config.TARGET] = self.target_encoder.transform(out[config.TARGET])
        return out

    # convenience
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # helpers for semesters
    @staticmethod
    def semester_features(df: pd.DataFrame, semesters: int) -> pd.DataFrame:
        """Return view with 0, 1 or 2 semester columns used in original notebook."""
        if semesters == 2:
            return df
        df = df.copy()
        second_sem = [c for c in df.columns if "2nd_sem" in c]
        first_sem = [c for c in df.columns if "1st_sem" in c]
        if semesters == 1:
            df.drop(columns=second_sem, inplace=True)
        elif semesters == 0:
            df.drop(columns=second_sem + first_sem, inplace=True)
        else:
            raise ValueError("semesters must be 0/1/2")
        return df