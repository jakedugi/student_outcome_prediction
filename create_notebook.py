import nbformat as nbf

nb = nbf.v4.new_notebook()

# Markdown cell with title and description
nb.cells.append(nbf.v4.new_markdown_cell("""# Student Outcome Prediction Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jakedugi/student_outcome_prediction/blob/main/demo.ipynb)

This notebook demonstrates the core functionality of our student outcome prediction model."""))

# Cell for Colab setup
nb.cells.append(nbf.v4.new_code_cell("""# Clone the repository if running in Colab
try:
    import google.colab
    !git clone https://github.com/jakedugi/student_outcome_prediction.git
    %cd student_outcome_prediction
except ImportError:
    pass  # Not running in Colab"""))

# Cell for installing dependencies
nb.cells.append(nbf.v4.new_code_cell("""# Install dependencies
!pip install -q -r requirements.txt
!pip install -q seaborn shap"""))

# Cell for imports
nb.cells.append(nbf.v4.new_code_cell("""import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import confusion_matrix
import numpy as np

from src.pipeline import TrainingPipeline
from src.config import TARGET"""))

# Markdown cell for training section
nb.cells.append(nbf.v4.new_markdown_cell("""## Training the Model

We'll train our model using data from the first 2 semesters:"""))

# Cell for training
nb.cells.append(nbf.v4.new_code_cell("""# Initialize and train pipeline
pipeline = TrainingPipeline()
results = pipeline.run(semesters=2)

# Get best model results
best_result = results[0]
print(f"Best model: {best_result['model']} with accuracy: {best_result['accuracy']:.3f}")"""))

# Markdown cell for visualization section
nb.cells.append(nbf.v4.new_markdown_cell("""## Model Performance Visualization

Let's visualize the confusion matrix to understand our model's performance:"""))

# Cell for confusion matrix
nb.cells.append(nbf.v4.new_code_cell("""def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Plot confusion matrix
plot_confusion_matrix(best_result['y_true'], best_result['y_pred'])"""))

# Markdown cell for feature importance section
nb.cells.append(nbf.v4.new_markdown_cell("""## Feature Importance Analysis

Let's examine which features are most important for prediction:"""))

# Cell for feature importance
nb.cells.append(nbf.v4.new_code_cell("""def plot_feature_importance(model, X, feature_names):
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(X.shape[1]), importances[indices])
        plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        # For other models, use SHAP values
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.show()

# Plot feature importance
plot_feature_importance(best_result['model_obj'], best_result['X_test'], best_result['feature_names'])"""))

# Write the notebook to a file
with open('demo.ipynb', 'w') as f:
    nbf.write(nb, f) 