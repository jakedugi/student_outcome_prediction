from __future__ import annotations
import pandas as pd
from pathlib import Path
from . import config, utils

logger = utils.logger


class DataLoader:
    """Simple csv to DataFrame loader."""

    def __init__(self, csv_path: str | Path | None = None) -> None:
        self.csv_path = Path(csv_path or config.DEFAULT_CSV)

    @utils.timer
    def load(self) -> pd.DataFrame:
        logger.info("Loading data from %s", self.csv_path)
        df = pd.read_csv(self.csv_path)
        df.columns = df.columns.str.replace(" ", "_")
        logger.info("Data shape: %s", df.shape)
        return df