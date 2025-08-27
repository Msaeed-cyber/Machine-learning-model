import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------
# This function loads the Iris dataset from sklearn.
# The dataset includes features of iris flowers (like petal length, width, etc.)
# and the target variable (species: Setosa, Versicolor, Virginica).
# Returns: features (X) and labels (y).
def load_data():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    return X, y


# ---------------------------------------------------
# 2. Cross-Validation Function
# ---------------------------------------------------
# This function performs Stratified K-Fold Cross Validation on the given model.
# - Model is trained and validated on multiple folds of the dataset.
# - It calculates multiple metrics: Accuracy, Precision, Recall, F1 Score.
# Returns: dictionary of cross-validation results including metrics.
def cross_validate_model(model, X, y, cv=5):
    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = cross_validate(model, X, y, cv=skf, scoring=scoring, return_estimator=True)
    return results


# ---------------------------------------------------
# 3. Compute Metrics Function
# ---------------------------------------------------
# This function trains the model on the dataset and predicts labels.
# It then computes a full classification report (precision, recall, f1-score, accuracy).
# Returns: classification report + predicted labels.
def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=True)
    return report, y_pred


# ---------------------------------------------------
# 4. Confusion Matrix Visualization
# ---------------------------------------------------
# This function plots a confusion matrix.
# A confusion matrix shows how many predictions were correct/incorrect
# for each class of the dataset.
def visualize_confusion_matrix(y, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.show()


# ---------------------------------------------------
# 5. ROC Curve (only for binary classification)
# ---------------------------------------------------
# This function plots the ROC curve for binary classification problems.
# - ROC curve shows trade-off between True Positive Rate and False Positive Rate.
# - AUC (Area Under Curve) gives an overall performance score.
# Note: Since Iris dataset has 3 classes, ROC curve will not work here.
def plot_roc_auc(model, X, y):
    if len(np.unique(y)) == 2:  # binary only
        y_score = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_score)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.show()
    else:
        print("ROC curve only available for binary classification.")


# ---------------------------------------------------
# 6. Main Pipeline
# ---------------------------------------------------
# This section executes the full ML pipeline step by step:
# - Load the dataset
# - Train and evaluate Logistic Regression using cross-validation
# - Print evaluation metrics (accuracy, precision, recall, F1-score)
# - Train a final model on the full dataset
# - Show classification report
# - Display confusion matrix
# - Plot ROC curve (only if binary classification dataset is used)
if __name__ == "__main__":
    # Load Data
    X, y = load_data()

    # Choose Model
    model = LogisticRegression(max_iter=200)

    # Cross-validation
    results = cross_validate_model(model, X, y, cv=5)

    print("Cross-validation Results:")
    for metric in ["test_accuracy", "test_precision_macro", "test_recall_macro", "test_f1_macro"]:
        print(f"{metric}: Mean={results[metric].mean():.4f}, Std={results[metric].std():.4f}")

    # Final Model (train on full data)
    final_model = LogisticRegression(max_iter=200)
    final_model.fit(X, y)

    # Compute metrics
    report, y_pred = compute_metrics(final_model, X, y)
    print("\nClassification Report:")
    print(pd.DataFrame(report).transpose())

    # Visualize Confusion Matrix
    visualize_confusion_matrix(y, y_pred)

    # ROC Curve (only for binary datasets)
    plot_roc_auc(final_model, X, y)
