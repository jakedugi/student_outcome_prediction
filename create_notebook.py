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
nb.cells.append(nbf.v4.new_markdown_cell("""## Model Performance Analysis

Let's analyze how well our model performs and understand what drives its predictions:

### 1. Prediction Accuracy by Student Outcome

First, let's look at how accurately our model predicts each type of student outcome:"""))

# Cell for confusion matrix
nb.cells.append(nbf.v4.new_code_cell("""def plot_confusion_matrix(y_true, y_pred, class_names=None):
    \"\"\"Plot confusion matrix with class labels.\"\"\"
    cm = confusion_matrix(y_true, y_pred)
    
    # Default class names if none provided
    if class_names is None:
        class_names = ['Dropped Out', 'Continuing Studies', 'Graduated']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Prediction Accuracy by Student Outcome', pad=20)
    plt.ylabel('True Outcome')
    plt.xlabel('Predicted Outcome')
    plt.tight_layout()
    plt.show()

# Plot confusion matrix with meaningful labels
plot_confusion_matrix(best_result['y_true'], best_result['y_pred'])"""))

# Add explanation for confusion matrix
nb.cells.append(nbf.v4.new_markdown_cell("""The confusion matrix above shows how well our model predicts each type of student outcome:

- Each row represents the **true outcome** for a group of students
- Each column shows what the model **predicted** for those students
- Numbers in each cell show how many students fall into each category
- Diagonal cells (top-left to bottom-right) show correct predictions
- Off-diagonal cells show where the model made mistakes

For example, if you look at the "Dropped Out" row, you can see:
- How many actual dropout students were correctly identified
- How many were incorrectly predicted to continue or graduate"""))

# Markdown cell for SHAP analysis
nb.cells.append(nbf.v4.new_markdown_cell("""### 2. Understanding What Drives Predictions

Now let's use SHAP (SHapley Additive exPlanations) values to understand:
- Which factors most strongly influence student outcomes
- How different features affect each type of outcome
- What patterns lead to different predictions"""))

# Cell for SHAP analysis
nb.cells.append(nbf.v4.new_code_cell("""# Calculate SHAP values
print("📊 Calculating feature importance using SHAP...")

def plot_shap_analysis(model, X, feature_names):
    \"\"\"Create SHAP summary and decision plots for model interpretation.\"\"\"
    # Clean up feature names for display
    display_names = [name.replace('_', ' ').title() for name in feature_names]
    
    try:
        # Create explainer
        if hasattr(model.estimator, 'predict_proba'):
            explainer = shap.TreeExplainer(model.estimator) if hasattr(model.estimator, 'apply') else shap.Explainer(model.estimator)
        else:
            explainer = shap.TreeExplainer(model.estimator) if hasattr(model.estimator, 'apply') else shap.Explainer(model.estimator)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)

        # Plot 1: Overall Summary Plot
        plt.figure(figsize=(14, 10))  # Increased figure size
        shap.summary_plot(
            shap_values,
            X,
            feature_names=display_names,
            show=False,
            plot_size=(12, 8)
        )
        plt.title("Feature Impact Across All Outcomes", pad=20)
        plt.tight_layout()
        plt.show()

        # Plot class-specific summary plots
        class_names = ['Dropout Risk', 'Continuing Studies', 'Graduation']
        for i, class_name in enumerate(class_names):
            plt.figure(figsize=(14, 10))
            shap.summary_plot(
                shap_values[i] if isinstance(shap_values, list) else shap_values[:,:,i],
                X,
                feature_names=display_names,
                show=False,
                plot_size=(12, 8)
            )
            plt.title(f"Feature Impact for {class_name}", pad=20)
            plt.tight_layout()
            plt.show()

        # Decision plots for each class
        expected_value = explainer.expected_value if hasattr(explainer, 'expected_value') else [0] * 3
        if isinstance(expected_value, (int, float)):
            expected_value = [expected_value]
            
        for i, class_name in enumerate(class_names):
            plt.figure(figsize=(12, 8))
            shap.decision_plot(
                expected_value[i] if isinstance(expected_value, list) else expected_value,
                shap_values[i] if isinstance(shap_values, list) else shap_values[:,:,i],
                X,
                feature_names=display_names,
                show=False
            )
            plt.title(f"Decision Plot for {class_name}", pad=20)
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"⚠️ Could not compute SHAP values: {str(e)}")
        print("Falling back to feature importances...")
        
        if hasattr(model.estimator, 'feature_importances_'):
            importances = model.estimator.feature_importances_
            indices = np.argsort(importances)[::-1][:15]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(x=range(len(indices)), 
                       y=importances[indices],
                       ax=ax,
                       palette='viridis')
            
            plt.title('Most Important Factors in Predicting Student Outcomes', pad=20)
            plt.xlabel('Factors')
            plt.ylabel('Importance Score')
            plt.xticks(range(len(indices)),
                      [display_names[i] for i in indices],
                      rotation=45,
                      ha='right')
            plt.tight_layout()
            plt.show()

# Generate SHAP analysis plots
plot_shap_analysis(
    best_result['model_obj'],
    best_result['X_test'],
    best_result['feature_names']
)"""))

# Add interpretation guidance
nb.cells.append(nbf.v4.new_markdown_cell("""### How to Interpret These Plots

1. **Overall Summary Plot**:
   - Shows feature importance across all outcomes
   - Features ordered by importance (top = most important)
   - Color indicates feature value (red = high, blue = low)
   - Width shows distribution of impact

2. **Class-Specific Summary Plots**:
   - One plot for each outcome (Dropout Risk, Continuing Studies, Graduation)
   - Shows how features specifically influence each outcome
   - Helps identify risk factors for dropping out vs. success factors for graduation

3. **Decision Plots**:
   - Shows how features combine to make predictions
   - Each line represents a student's prediction path
   - Start at base value (left), end at final prediction (right)
   - Steeper slopes = stronger feature impact
   - Direction shows whether feature increases/decreases likelihood

This analysis helps identify:
- Early warning signs of dropout risk
- Key factors promoting student success
- How different features interact to influence outcomes
- Where to focus intervention efforts"""))

# Write the notebook to a file
with open('demo.ipynb', 'w') as f:
    nbf.write(nb, f) 
