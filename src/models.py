"""Classifiers and evaluation metrics used in the study."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_classifiers(random_state: int = 42) -> dict:
    """The six classifiers compared in the report (hyper-parameters as tuned)."""
    return {
        "Logistic Regression": LogisticRegression(C=100, max_iter=5000),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "ANN (MLP)": MLPClassifier(hidden_layer_sizes=(16, 8), alpha=1e-2,
                                   max_iter=3000, random_state=random_state),
        "SVM (Linear)": SVC(kernel="linear", C=100, gamma=0.009, random_state=random_state),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=random_state),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, precision, sensitivity (recall), specificity, F1 and MCC."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
