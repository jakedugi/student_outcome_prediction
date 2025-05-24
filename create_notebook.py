import nbformat as nbf
import os

# Create data directory and .gitkeep
os.makedirs("data", exist_ok=True)
with open("data/.gitkeep", "w") as f:
    pass

nb = nbf.v4.new_notebook()

# Markdown cell with title and description
nb.cells.append(nbf.v4.new_markdown_cell("""# Student Outcome Prediction Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jakedugi/student_outcome_prediction/blob/main/demo.ipynb)

This notebook demonstrates the core functionality of our student outcome prediction model."""))

# Add Kaggle dataset setup instructions
nb.cells.append(nbf.v4.new_markdown_cell("""## ⚠️ Dataset Setup

This demo uses data from [Kaggle](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention). To run it, you'll need to:

1. Go to [Kaggle.com](https://www.kaggle.com) → Account → Create API Token
2. Download your `kaggle.json` file
3. Upload it when prompted below

> 💡 This is a one-time setup. Your API key will be stored securely."""))

# Cell for Colab setup and Kaggle authentication
nb.cells.append(nbf.v4.new_code_cell("""# Clone the repository if running in Colab
try:
    import google.colab
    !git clone https://github.com/jakedugi/student_outcome_prediction.git
    %cd student_outcome_prediction
except ImportError:
    pass  # Not running in Colab

# Install required packages
!pip install -q kaggle

# Ensure data directory exists
import os
os.makedirs("data", exist_ok=True)"""))

# Cell for Kaggle authentication
nb.cells.append(nbf.v4.new_code_cell("""# Set up Kaggle credentials
import os
from pathlib import Path

def setup_kaggle_credentials():
    try:
        from google.colab import files
        print("📤 Please upload your kaggle.json file...")
        uploaded = files.upload()
        
        if not uploaded:
            raise Exception("No file was uploaded")
            
        # Create Kaggle directory and move credentials
        !mkdir -p ~/.kaggle
        !cp kaggle.json ~/.kaggle/
        !chmod 600 ~/.kaggle/kaggle.json
        print("✅ Kaggle credentials configured successfully!")
        
    except ImportError:
        # Local environment - check if credentials exist
        kaggle_path = Path.home() / '.kaggle' / 'kaggle.json'
        if not kaggle_path.exists():
            print("⚠️ Please place your kaggle.json in:", kaggle_path)
            return False
        print("✅ Found existing Kaggle credentials")
    return True

if setup_kaggle_credentials():
    print("\\n🔄 Downloading dataset...")
    try:
        !kaggle datasets download -d thedevastator/higher-education-predictors-of-student-retention --quiet
        !unzip -q higher-education-predictors-of-student-retention.zip -d data/
        !rm higher-education-predictors-of-student-retention.zip
        print("✅ Dataset downloaded and extracted to data/")
    except Exception as e:
        print("❌ Failed to download dataset:", str(e))
        print("⚠️ Please download manually from: https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention")"""))

# Cell for installing dependencies
nb.cells.append(nbf.v4.new_code_cell("""# Install remaining dependencies
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
