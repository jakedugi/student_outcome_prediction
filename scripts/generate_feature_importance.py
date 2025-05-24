#!/usr/bin/env python
"""Generate feature importance plot for the best model."""
from pathlib import Path
from src.pipeline import TrainingPipeline
from src.visualization import plot_feature_importance
from src import config

def main():
    # Create reports directory
    reports_dir = config.ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Train models and get best result
    pipeline = TrainingPipeline()
    results = pipeline.run(semesters=2)
    best_result = results[0]
    
    # Generate and save plot
    plot_feature_importance(
        model=best_result['model_obj'],
        X=best_result['X_test'],
        output_path=reports_dir / "feature_importance.png"
    )

if __name__ == "__main__":
    main() 