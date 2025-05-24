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
nb.cells.append(nbf.v4.new_code_cell("""# Install and import required packages
!pip install -q seaborn shap

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import confusion_matrix
import numpy as np

from src.pipeline import TrainingPipeline
from src.config import TARGET

# Apply Seaborn's modern styling
sns.set_theme(style='whitegrid', font_scale=1.1)  # Slightly larger fonts for better readability"""))

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

Let's examine which factors most strongly influence student outcomes. This analysis helps us understand:
- What predicts student success
- Where to focus intervention efforts
- Early warning signs of dropout risk"""))

# Cell for feature importance
nb.cells.append(nbf.v4.new_code_cell("""# Define class descriptions for better readability
class_descriptions = {
    0: "Dropout Risk",
    1: "Continuing Studies", 
    2: "Likely to Graduate"
}

def plot_feature_importance(model_wrapper, X, feature_names, max_display=15):
    \"\"\"Plot feature importance with improved visualization and explanations.\"\"\"
    # Clean up feature names for display
    display_names = [name.replace('_', ' ').title() for name in feature_names]
    
    if hasattr(model_wrapper.estimator, 'feature_importances_'):
        # For tree-based models (Random Forest, XGBoost, etc.)
        importances = model_wrapper.estimator.feature_importances_
        indices = np.argsort(importances)[::-1][:max_display]
        
        # Create figure with seaborn style
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x=range(len(indices)), y=importances[indices], ax=ax, color='cornflowerblue')
        
        # Add value labels on top of bars
        for i, bar in enumerate(ax.patches):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom')
        
        plt.title(f'Feature Impact on Student Outcomes\\n{model_wrapper.model_name}', 
                 pad=20, wrap=True)
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.xticks(range(len(indices)), 
                  [display_names[i] for i in indices],
                  rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        try:
            # For models without feature_importances_, use SHAP
            predict_fn = lambda x: model_wrapper.estimator.predict_proba(x)[:, 1] if hasattr(model_wrapper.estimator, 'predict_proba') else model_wrapper.estimator.predict
            
            explainer = shap.Explainer(predict_fn, X)
            shap_values = explainer(X)
            
            # Handle different SHAP value shapes
            if isinstance(shap_values, shap.Explanation):
                if len(shap_values.shape) > 2:  # Multiclass case
                    # Plot for all classes
                    for class_idx in range(shap_values.shape[2]):
                        plt.figure(figsize=(12, 8))
                        class_name = class_descriptions.get(class_idx, f"Class {class_idx}")
                        
                        # Custom summary plot with better formatting
                        shap.summary_plot(
                            shap_values[:, :, class_idx],
                            X,
                            feature_names=display_names,
                            plot_type="bar",
                            max_display=max_display,
                            show=False,
                            plot_size=(12, 8)
                        )
                        
                        plt.title(f'Feature Impact on {class_name} Outcome\\n{model_wrapper.model_name}',
                                pad=20, wrap=True)
                        plt.xlabel('Average Impact on Prediction (SHAP Value)')
                        
                        # Add legend explaining SHAP values
                        plt.figtext(1.02, 0.5, 
                                  'How to read this plot:\\n\\n' +
                                  '• Longer bars = Stronger impact\\n' +
                                  '• Red = Higher feature values\\n' +
                                  '• Blue = Lower feature values\\n' +
                                  '• Values show average impact\\n' +
                                  '  on model predictions',
                                  fontsize=10, ha='left', va='center',
                                  bbox=dict(facecolor='white', alpha=0.8))
                        
                        plt.tight_layout()
                        plt.show()
                else:  # Binary classification or regression
                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(
                        shap_values,
                        X,
                        feature_names=display_names,
                        plot_type="bar",
                        max_display=max_display,
                        show=False
                    )
                    plt.title(f'Feature Impact on Student Outcomes\\n{model_wrapper.model_name}',
                            pad=20, wrap=True)
                    plt.xlabel('Average Impact on Prediction (SHAP Value)')
                    plt.tight_layout()
                    plt.show()
        except Exception as e:
            print(f"⚠️ Could not compute SHAP values: {str(e)}")
            print("Falling back to coefficients if available...")
            
            if hasattr(model_wrapper.estimator, 'coef_'):
                coef = model_wrapper.estimator.coef_
                coef = coef if len(coef.shape) == 1 else coef[0]
                importance = np.abs(coef)
                indices = np.argsort(importance)[::-1][:max_display]
                
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.barplot(x=range(len(indices)), y=importance[indices], ax=ax, color='cornflowerblue')
                plt.title(f'Feature Impact on Student Outcomes\\n{model_wrapper.model_name}',
                         pad=20, wrap=True)
                plt.xlabel('Features')
                plt.ylabel('Absolute Coefficient Value')
                plt.xticks(range(len(indices)), 
                          [display_names[i] for i in indices],
                          rotation=45, ha='right')
                plt.tight_layout()
                plt.show()

print("📊 Analyzing what influences student outcomes...")
plot_feature_importance(
    best_result['model_obj'],
    best_result['X_test'],
    best_result['feature_names']
)"""))

# Add explanation of the results
nb.cells.append(nbf.v4.new_markdown_cell("""### Key Insights from Feature Importance

The plot above shows which factors most strongly influence student outcomes. Here's how to interpret it:

1. **Bar Length**: Longer bars indicate stronger influence on predictions
2. **Colors** (for SHAP plots):
   - Red = Higher values of that feature
   - Blue = Lower values of that feature
3. **Direction**:
   - Positive values (right) increase likelihood of the outcome
   - Negative values (left) decrease likelihood

This analysis helps identify early warning signs and potential intervention points to improve student success."""))

# Write the notebook to a file
with open('demo.ipynb', 'w') as f:
    nbf.write(nb, f) 
