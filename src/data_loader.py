from __future__ import annotations
import pandas as pd
import os
from pathlib import Path
from . import config, utils

logger = utils.logger


class DataLoader:
    """Simple csv to DataFrame loader"""

    def __init__(self, csv_path: str | Path | None = None) -> None:
        self.csv_path = Path(csv_path or config.DEFAULT_CSV)

    @utils.timer
    def load(self) -> pd.DataFrame:
        # Ensure data directory exists
        os.makedirs(config.DATA_DIR, exist_ok=True)
        
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.csv_path}. "
                "Please download it from Kaggle and place it in the data/ directory."
            )
            
        logger.info("Loading data from %s", self.csv_path)
        df = pd.read_csv(self.csv_path)
        df.columns = df.columns.str.replace(" ", "_")
        logger.info("Data shape: %s", df.shape)
        return df