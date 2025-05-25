"""Configuration management for the student outcome prediction project."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Dict, Any, Optional
import yaml
import logging

# paths
ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT / "data"
DEFAULT_CSV: Path = DATA_DIR / "dataset.csv"
ARTIFACT_DIR: Path = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

# columns 
NUMERIC_COLUMNS: Sequence[str] = [
    "Application_order",
    "Age_at_enrollment",
    # 1st semester
    "Curricular_units_1st_sem_(credited)",
    "Curricular_units_1st_sem_(enrolled)",
    "Curricular_units_1st_sem_(evaluations)",
    "Curricular_units_1st_sem_(approved)",
    "Curricular_units_1st_sem_(grade)",
    "Curricular_units_1st_sem_(without_evaluations)",
    # 2nd semester
    "Curricular_units_2nd_sem_(credited)",
    "Curricular_units_2nd_sem_(enrolled)",
    "Curricular_units_2nd_sem_(evaluations)",
    "Curricular_units_2nd_sem_(approved)",
    "Curricular_units_2nd_sem_(grade)",
    "Curricular_units_2nd_sem_(without_evaluations)",
]

TARGET: str = "Target"
TEST_SIZE: float = 0.20
RANDOM_STATE: int = 42


# dataclasses
@dataclass(frozen=True)
class TrainSettings:
    """Hyper-parameters that might change between runs"""
    test_size: float = TEST_SIZE
    random_state: int = RANDOM_STATE
    models: tuple[str, ...] = field(
        default_factory=lambda: (
            "decision_tree",
            "gradient_boosting",
            "random_forest",
            "xgboost",
            "log_reg",
            "svm",
            "knn",
            "adaboost",
            "qda",
            "naive_bayes",
            "neural_net",
        )
    )

@dataclass
class LoggingConfig:
    """Logging configuration settings.
    
    Attributes:
        level: Logging level (DEBUG, INFO, etc)
        format: Log message format string
        file: Optional log file path
    """
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None

@dataclass
class ModelConfig:
    """Model training configuration.
    
    Attributes:
        random_state: Random seed for reproducibility
        test_size: Fraction of data to use for testing
        cv_folds: Number of cross-validation folds
    """
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5

@dataclass
class VisualizationConfig:
    """Visualization settings.
    
    Attributes:
        style: Matplotlib style to use
        dpi: Figure DPI
        save_format: Default format for saving plots
    """
    style: str = "seaborn"
    dpi: int = 300
    save_format: str = "png"

@dataclass
class Config:
    """Main configuration class.
    
    Attributes:
        logging: Logging settings
        model: Model training settings
        viz: Visualization settings
        data_dir: Path to data directory
        output_dir: Path to output directory
    """
    logging: LoggingConfig
    model: ModelConfig
    viz: VisualizationConfig
    data_dir: Path
    output_dir: Path

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        """Load configuration from YAML file.
        
        Args:
            path: Path to YAML config file
            
        Returns:
            Loaded configuration object
            
        Raises:
            ValueError: If config is invalid
        """
        with open(path) as f:
            data = yaml.safe_load(f)
            
        return cls(
            logging=LoggingConfig(**data.get("logging", {})),
            model=ModelConfig(**data.get("model", {})),
            viz=VisualizationConfig(**data.get("viz", {})),
            data_dir=Path(data["data_dir"]),
            output_dir=Path(data["output_dir"])
        )

    def validate(self) -> None:
        """Validate configuration settings.
        
        Raises:
            ValueError: If any settings are invalid
        """
        # Validate logging
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.logging.level not in valid_levels:
            raise ValueError(f"Invalid logging level: {self.logging.level}")
            
        # Validate model settings
        if not 0 < self.model.test_size < 1:
            raise ValueError(f"Invalid test size: {self.model.test_size}")
        if self.model.cv_folds < 2:
            raise ValueError(f"Invalid CV folds: {self.model.cv_folds}")
            
        # Validate paths
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup_logging(self) -> None:
        """Configure logging based on settings."""
        logging.basicConfig(
            level=getattr(logging, self.logging.level),
            format=self.logging.format,
            filename=self.logging.file
        )

# Global config instance
config: Optional[Config] = None

def load_config(path: str) -> Config:
    """Load and validate configuration.
    
    Args:
        path: Path to config file
        
    Returns:
        Validated config object
    """
    global config
    config = Config.from_yaml(path)
    config.validate()
    config.setup_logging()
    return config

def get_config() -> Config:
    """Get the current configuration.
    
    Returns:
        Current config object
        
    Raises:
        RuntimeError: If config not initialized
    """
    if config is None:
        raise RuntimeError("Configuration not initialized. Call load_config first.")
    return config