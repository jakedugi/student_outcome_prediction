"""Library Import"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
# %matplotlib inline

"""Import Data Set - https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention/data """

#Jupyter version
#file_path = 'studentkaggledataset.csv'
#dataset = pd.read_csv(file_path)

#Colab version, Uploaded .csv in Google Drive
from google.colab import drive
drive.mount('/content/drive')

#Read into Pandas DataFrame
dataset = pd.read_csv(r'/content/drive/MyDrive/studentkaggledataset.csv')
#reading excel file to convert into dataframe
#print(dataset)

"""Inspect Data & Column Headers:"""

dataset.shape
dataset.head()

print("\nThe column headers :")
print("Column headers from list(df):", list(dataset))

#Python readability
def replace_spaces_with_underscores(dataset):
    dataset.columns = dataset.columns.str.replace(' ', '_')
    item_counts = dataset.value_counts()
    print(item_counts)
    print(dataset)

replace_spaces_with_underscores(dataset)

"""Standardize & Normalize Numerical Data"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler

#Specify the numerical columns to be scaled
columns_to_scale = ['Application_order', 'Age_at_enrollment', 'Curricular_units_1st_sem_(credited)',
                    'Curricular_units_1st_sem_(enrolled)', 'Curricular_units_1st_sem_(evaluations)',
                    'Curricular_units_1st_sem_(approved)', 'Curricular_units_1st_sem_(grade)',
                    'Curricular_units_1st_sem_(without_evaluations)', 'Curricular_units_2nd_sem_(credited)',
                    'Curricular_units_2nd_sem_(enrolled)', 'Curricular_units_2nd_sem_(evaluations)',
                    'Curricular_units_2nd_sem_(approved)', 'Curricular_units_2nd_sem_(grade)', 'Curricular_units_2nd_sem_(without_evaluations)']

#Create a copy with only the numerical columns
df_selected = dataset[columns_to_scale].copy()

#Create a StandardScaler object
scaler_standard = StandardScaler()

#Fit and transform the selected columns standardizing
df_selected_scaled = scaler_standard.fit_transform(df_selected)

#Store the mapping and scaler of standardization
scaler_standard_mapping = {col: (scaler_standard.mean_[i], scaler_standard.scale_[i]) for i, col in enumerate(columns_to_scale)}
scaler_standard_info = {'scaler': scaler_standard, 'mapping': scaler_standard_mapping}

#Replace the scaled columns in the original dataset
df_scaled = dataset.copy()
df_scaled[columns_to_scale] = df_selected_scaled

#Create a MinMaxScaler object
scaler_minmax = MinMaxScaler()

#Fit and transform the selected columns normalizing
df_selected_normalized = scaler_minmax.fit_transform(df_selected)

#Store the mapping and scaler of normalization
scaler_minmax_mapping = {col: (scaler_minmax.data_min_[i], scaler_minmax.data_max_[i]) for i, col in enumerate(columns_to_scale)}
scaler_minmax_info = {'scaler': scaler_minmax, 'mapping': scaler_minmax_mapping}

#Replace the normalized columns in the original dataset
df_normalized = dataset.copy()
df_normalized[columns_to_scale] = df_selected_normalized

#Standardized and Normalized df
df = df_normalized

"""Accessing Mapping"""

print("StandardScaler Standardization Mapping:")
for col, (mean, scale) in scaler_standard_info['mapping'].items():
    print(f"Column '{col}': Mean={mean}, Scale={scale}")

print("MinMaxScaler Normalization Mapping:")
for col, (data_min, data_max) in scaler_minmax_info['mapping'].items():
    print(f"Column '{col}': Data Min={data_min}, Data Max={data_max}")

"""Feature Target & Test Training Split:"""

#Input features = X everthying but, Target = y
y = df[['Target']]
X = df.drop(['Target'], axis=1)
print(X)
print(y)

#Splitting 20 percent into test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 42)

X_train.info()
X_test.info()
y_train.info()
y_test.info()

"""Categorical Encoding 'Target'"""

from sklearn.preprocessing import LabelEncoder

def label_encode_columns(train_df, test_df):
    #Get the object dtype columns
    object_cols = train_df.select_dtypes(include=['object']).columns

    #Combine the train and test dfs
    combined_df = pd.concat([train_df[object_cols], test_df[object_cols]])

    #Initialize a dictionary
    mapping = {}

    #Fit the label encoder on the data
    for col in object_cols:
        label_encoder = LabelEncoder()
        combined_df[col] = label_encoder.fit_transform(combined_df[col])
        mapping[col] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        train_df[col] = label_encoder.transform(train_df[col])
        test_df[col] = label_encoder.transform(test_df[col])

    #Return the transformed train and test dataframes with mapping
    return train_df, test_df, mapping

#Call function
y_train, y_test, mapping = label_encode_columns(y_train, y_test)

print(mapping)

y_train.dtypes
y_train.head()

"""Models"""

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report
import graphviz

def classify_with_decision_tree(X_train, y_train, X_test, y_test, target_column, plot_number):
    # Create decision tree classifier object
    decision_tree = DecisionTreeClassifier()

    # Fit classifier on training data
    decision_tree.fit(X_train, y_train[target_column])

    # Make predictions on test data
    y_pred = decision_tree.predict(X_test)

    # Print classification report for target column
    print(f"Classification report for column '{target_column}':")
    print(classification_report(y_test[target_column].astype(str), y_pred.astype(str)))

    # Plot decision tree
    dot_data = export_graphviz(decision_tree, out_file=None,
                               feature_names=X_train.columns,
                               class_names=y_train[target_column].astype(str).unique(),
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render(f"decision_tree_{plot_number}")

    # Return the predictions
    return y_pred

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def classify_with_gradient_boosting_classifier(X_train, y_train, X_test, y_test, target_column, plot_number):
    # Create gradient boosting classifier object
    gradient_boosting = GradientBoostingClassifier()

    # Fit classifier on training data
    gradient_boosting.fit(X_train, y_train[target_column])

    # Make predictions on test data
    y_pred = gradient_boosting.predict(X_test)

    # Print classification report for target column
    print(f"Classification report for column '{target_column}':")
    print(classification_report(y_test[target_column].astype(str), y_pred.astype(str)))

    # Plot feature importance
    feature_importance = gradient_boosting.feature_importances_
    sorted_idx = feature_importance.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(X_train.columns[sorted_idx], feature_importance[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title(f'Feature Importance Plot ({plot_number})')
    plt.tight_layout()
    plt.savefig(f"gbc_feature_importance_plot_{plot_number}.png")

    # Return the predictions and feature importance plot
    return y_pred, feature_importance

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def classify_with_random_forest(X_train, y_train, X_test, y_test, target_column):
    # Create random forest classifier object
    random_forest = RandomForestClassifier(n_estimators=100)

    # Fit classifier on training data
    random_forest.fit(X_train, y_train[target_column])

    # Make predictions on test data
    y_pred = random_forest.predict(X_test)

    # Print classification report for target column
    print(f"Classification report for column '{target_column}':")
    print(classification_report(y_test[target_column], y_pred))

    # Return the predictions
    return y_pred

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def classify_with_boosted_random_forest(X_train, y_train, X_test, y_test, target_column):
    # Create random forest classifier object
    random_forest = RandomForestClassifier(n_estimators=100)

    # Create gradient boosting classifier object
    boosting_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)

    # Fit boosted random forest on training data
    boosting_classifier.fit(X_train, y_train[target_column])

    # Make predictions on test data
    y_pred = boosting_classifier.predict(X_test)

    # Print classification report for target column
    print(f"Classification report for column '{target_column}':")
    print(classification_report(y_test[target_column], y_pred))

    # Return the predictions
    return y_pred

from sklearn.svm import SVC
from sklearn.metrics import classification_report

def classify_with_svm(X_train, y_train, X_test, y_test, target_column):
    # Create SVM classifier object
    svm = SVC()

    # Fit classifier on training data
    svm.fit(X_train, y_train[target_column])

    # Make predictions on test data
    y_pred = svm.predict(X_test)

    # Print classification report for target column
    print(f"Classification report for column '{target_column}':")
    print(classification_report(y_test[target_column], y_pred,zero_division=1))

    # Return the predictions
    return y_pred

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

def create_nn_model(input_dim, num_classes):
    # Create neural network model with fixed hyperparameters
    model = Sequential()
    model.add(Dense(units=16, input_dim=input_dim, activation='relu'))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))  # Use softmax for multi-class classification

    # Compile model with Adam optimizer and fixed learning rate
    optimizer = Adam(learning_rate=0.001)  # Use learning_rate instead of lr
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def classify_with_neural_network(X_train, y_train, X_test, y_test, target_column):
    # Encode target column into numerical labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train[target_column])
    y_test_encoded = label_encoder.transform(y_test[target_column])

    # Create neural network model
    input_dim = X_train.shape[1]
    num_classes = len(label_encoder.classes_)
    model = create_nn_model(input_dim, num_classes)

    # Fit model on training data
    model.fit(X_train, y_train_encoded, epochs=20, batch_size=32, verbose=0)

    # Make predictions on test data
    y_pred_proba = model.predict(X_test)
    y_pred = y_pred_proba.argmax(axis=1)  # Convert softmax probabilities to class labels

    # Print classification report for target column
    print(f"Classification report for column '{target_column}':")
    print(classification_report(y_test_encoded, y_pred))

    # Return the predictions
    return y_pred

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

def classify_with_naive_bayes(X_train, y_train, X_test, y_test, target_column):
    # Create a Gaussian Naive Bayes classifier object
    naive_bayes = GaussianNB()

    # Fit classifier on training data
    naive_bayes.fit(X_train, y_train[target_column])

    # Make predictions on test data
    y_pred = naive_bayes.predict(X_test)

    # Print classification report for target column
    print(f"Classification report for column '{target_column}':")
    print(classification_report(y_test[target_column].astype(str), y_pred.astype(str)))

    # Return the predictions
    return y_pred

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def classify_with_logistic_regression(X_train, y_train, X_test, y_test, target_column, plot_number, top_n=5):
    # Create logistic regression object
    logreg = LogisticRegression(max_iter=10000)

    # Fit classifier on training data
    logreg.fit(X_train, y_train[target_column])

    # Make predictions on test data
    y_pred = logreg.predict(X_test)

    # Print classification report for target column
    print(f"Classification report for column '{target_column}':")
    print(classification_report(y_test[target_column].astype(str), y_pred.astype(str)))

    # Get feature names
    feature_names = X_train.columns

    # Get absolute values of coefficients and their indices
    abs_coefficients = abs(logreg.coef_[0])
    top_indices = abs_coefficients.argsort()[-top_n:]

    # Extract top N coefficients and feature names
    top_coefficients = logreg.coef_[0][top_indices]
    top_features = feature_names[top_indices]

    # Plot top feature coefficients
    plt.figure(figsize=(10, 6))
    plt.barh(top_features, top_coefficients)
    plt.xlabel('Coefficient Magnitude')
    plt.ylabel('Features')
    plt.title(f'Top {top_n} Feature Coefficients Plot ({plot_number})')
    plt.tight_layout()
    plt.savefig(f"logit_top_feature_coefficients_plot_{plot_number}.png")

    # Return the predictions
    return y_pred

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def classify_with_knn_grid_search(X_train, y_train, X_test, y_test, target_column):
    # Define the parameter grid for grid search
    param_grid = {'n_neighbors': [7, 11, 13, 15, 17, 19]}  # Define the range of n_neighbors values to explore

    # Create KNN classifier object
    knn = KNeighborsClassifier()

    # Perform grid search with 5-fold cross-validation
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train[target_column])

    # Get the best estimator from grid search
    best_knn = grid_search.best_estimator_

    # Make predictions on test data using the best estimator
    y_pred = best_knn.predict(X_test)

    # Print best parameters and classification report for target column
    print("Best parameters found during grid search:")
    print(grid_search.best_params_)
    print("\nClassification report for column '{target_column}':")
    print(classification_report(y_test[target_column].astype(str), y_pred.astype(str)))

    # Return the predictions
    return y_pred

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

def classify_with_adaboost(X_train, y_train, X_test, y_test, target_column):
    # Get target column index
    target_col_idx = list(y_train.columns).index(target_column)

    # Get target column from train and test sets
    y_train = y_train.iloc[:, target_col_idx]
    y_test = y_test.iloc[:, target_col_idx]

    # Train Adaboost model with Decision Tree as base estimator
    model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=10), n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Print classification report for target column
    print(f"Classification Report for {target_column}:")
    print(classification_report(y_test, y_pred))

    return model

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report

def classify_with_qda(X_train, y_train, X_test, y_test, target_column):
    # Initialize QDA classifier
    qda = QuadraticDiscriminantAnalysis()

    # Train the model using the training data
    qda.fit(X_train, y_train[target_column])

    # Make predictions on test data
    y_pred = qda.predict(X_test)

    # Print classification report for target column
    print(f"Classification Report for {target_column}:")
    print(classification_report(y_test[target_column], y_pred))

from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def classify_with_xgboost(X_train, y_train, X_test, y_test, target_column, plot_number):
    # Get target column from train and test sets
    y_train = y_train[target_column]
    y_test = y_test[target_column]

    # Train XGBoost model
    model = XGBClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Print classification report for target column
    print(f"Classification Report for {target_column}:")
    print(classification_report(y_test, y_pred))

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plot_importance(model, max_num_features=10)  # Adjust max_num_features as needed
    plt.title(f'Feature Importance Plot ({plot_number})')
    plt.tight_layout()
    plt.savefig(f"xgb_feature_importance_plot_{plot_number}.png")

    return model

"""Training and Making Predictions 2 semesters of data"""

classify_with_decision_tree(X_train, y_train, X_test, y_test,'Target', plot_number = 2)

classify_with_gradient_boosting_classifier(X_train, y_train, X_test, y_test, 'Target', plot_number=2)

classify_with_random_forest(X_train, y_train, X_test, y_test, 'Target')

classify_with_boosted_random_forest(X_train, y_train, X_test, y_test, 'Target')

classify_with_svm(X_train, y_train, X_test, y_test, 'Target')

classify_with_neural_network(X_train, y_train, X_test, y_test, 'Target')

classify_with_naive_bayes(X_train, y_train, X_test, y_test, 'Target')

classify_with_logistic_regression(X_train, y_train, X_test, y_test, 'Target', plot_number=2, top_n=5)

classify_with_knn_grid_search(X_train, y_train, X_test, y_test, "Target")

classify_with_adaboost(X_train, y_train, X_test, y_test, 'Target')

classify_with_qda(X_train, y_train, X_test, y_test, 'Target')

classify_with_xgboost(X_train, y_train, X_test, y_test, 'Target', plot_number = 2)

"""Creating dataset with 1 semesters of data"""

#Initialize empty DataFrames
X_train_1_semesters = pd.DataFrame()
X_test_1_semesters = pd.DataFrame()

#Copy DataFrames
X_train_1_semesters = X_train.copy()
X_test_1_semesters = X_test.copy()

#Remove columns from X_test_1_semesters and X_train_1_semesters
cols_to_remove_1 = ['Curricular_units_2nd_sem_(credited)', 'Curricular_units_2nd_sem_(enrolled)', 'Curricular_units_2nd_sem_(evaluations)', 'Curricular_units_2nd_sem_(approved)', 'Curricular_units_2nd_sem_(grade)', 'Curricular_units_2nd_sem_(without_evaluations)']
X_test_1_semesters = X_test_1_semesters.drop(cols_to_remove_1, axis=1)
X_train_1_semesters = X_train_1_semesters.drop(cols_to_remove_1, axis=1)

#Display the heads of DataFrames
print(X_train_1_semesters.head())
print(X_test_1_semesters.head())

"""Training and Making Predictions 1 semester of data"""

classify_with_decision_tree(X_train_1_semesters, y_train, X_test_1_semesters, y_test, 'Target', plot_number=1)

classify_with_gradient_boosting_classifier(X_train_1_semesters, y_train, X_test_1_semesters, y_test, 'Target', plot_number=1)

classify_with_random_forest(X_train_1_semesters, y_train, X_test_1_semesters, y_test, 'Target')

classify_with_boosted_random_forest(X_train_1_semesters, y_train, X_test_1_semesters, y_test, 'Target')

classify_with_svm(X_train_1_semesters, y_train, X_test_1_semesters, y_test, 'Target')

classify_with_neural_network(X_train_1_semesters, y_train, X_test_1_semesters, y_test, 'Target')

classify_with_naive_bayes(X_train_1_semesters, y_train, X_test_1_semesters, y_test, 'Target')

classify_with_logistic_regression(X_train_1_semesters, y_train, X_test_1_semesters, y_test, 'Target', plot_number=1, top_n=5)

classify_with_knn_grid_search(X_train_1_semesters, y_train, X_test_1_semesters, y_test, "Target")

classify_with_adaboost(X_train_1_semesters, y_train, X_test_1_semesters, y_test, 'Target')

classify_with_qda(X_train_1_semesters, y_train, X_test_1_semesters, y_test, 'Target')

classify_with_xgboost(X_train_1_semesters, y_train, X_test_1_semesters, y_test, 'Target', plot_number = 1)

"""Creating dataset with 0 semesters of data"""

#Initialize empty DataFrames
X_train_0_semesters = pd.DataFrame()
X_test_0_semesters = pd.DataFrame()

#Copy DataFrames
X_train_0_semesters = X_train.copy()
X_test_0_semesters = X_test.copy()

#Remove columns from X_test_0_semesters and X_train_0_semesters
cols_to_remove_0 = ['Curricular_units_1st_sem_(credited)', 'Curricular_units_1st_sem_(enrolled)', 'Curricular_units_1st_sem_(evaluations)', 'Curricular_units_1st_sem_(approved)', 'Curricular_units_1st_sem_(grade)', 'Curricular_units_1st_sem_(without_evaluations)', 'Curricular_units_2nd_sem_(credited)', 'Curricular_units_2nd_sem_(enrolled)', 'Curricular_units_2nd_sem_(evaluations)', 'Curricular_units_2nd_sem_(approved)', 'Curricular_units_2nd_sem_(grade)', 'Curricular_units_2nd_sem_(without_evaluations)']
X_test_0_semesters = X_test_0_semesters.drop(cols_to_remove_0, axis=1)
X_train_0_semesters = X_train_0_semesters.drop(cols_to_remove_0, axis=1)

#Display the heads of DataFrames
print(X_train_0_semesters.head())
print(X_test_0_semesters.head())

"""Training and Making Predictions 0 semesters of data new admits"""

classify_with_decision_tree(X_train_0_semesters, y_train, X_test_0_semesters, y_test, 'Target', plot_number=0)

classify_with_gradient_boosting_classifier(X_train_0_semesters, y_train, X_test_0_semesters, y_test, 'Target',plot_number=0)

classify_with_random_forest(X_train_0_semesters, y_train, X_test_0_semesters, y_test, 'Target')

classify_with_boosted_random_forest(X_train_0_semesters, y_train, X_test_0_semesters, y_test, 'Target')

classify_with_svm(X_train_0_semesters, y_train, X_test_0_semesters, y_test, 'Target')

classify_with_neural_network(X_train_0_semesters, y_train, X_test_0_semesters, y_test, 'Target')

classify_with_naive_bayes(X_train_0_semesters, y_train, X_test_0_semesters, y_test, 'Target')

classify_with_logistic_regression(X_train_0_semesters, y_train, X_test_0_semesters, y_test, 'Target', plot_number=0, top_n=5)

classify_with_knn_grid_search(X_train_0_semesters, y_train, X_test_0_semesters, y_test, "Target")

classify_with_adaboost(X_train_0_semesters, y_train, X_test_0_semesters, y_test, 'Target')

classify_with_qda(X_train_0_semesters, y_train, X_test_0_semesters, y_test, 'Target')

classify_with_xgboost(X_train_0_semesters, y_train, X_test_0_semesters, y_test, 'Target', plot_number = 0)