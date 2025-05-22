"""
Central place to register every model available to the pipeline.
Add new entries in one line.  Order controls the training log order.
"""
from .sklearn_wrappers import (
    DecisionTree,
    GradientBoosting,
    RandomForest,
    XGBoost,
    LogisticReg,
    SVM,
    KNN,
    AdaBoost,
    QDA,
    NaiveBayes,
)
from .neural_net import NeuralNet

MODEL_REGISTRY = {
    "decision_tree": DecisionTree,
    "gradient_boosting": GradientBoosting,
    "random_forest": RandomForest,
    "xgboost": XGBoost,
    "log_reg": LogisticReg,
    "svm": SVM,
    "knn": KNN,
    "adaboost": AdaBoost,
    "qda": QDA,
    "naive_bayes": NaiveBayes,
    "neural_net": NeuralNet,
}