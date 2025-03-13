import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(relative_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame using a path relative to this script.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, relative_path)
    print("Loading data from:", full_path)
    return pd.read_csv(full_path)

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate a model on the test set and return performance metrics.
    """
    y_pred = model.predict(X_test)
    # Use predict_proba for ROC-AUC if available
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0)
    }
    return metrics

def main():
    # Load the processed data.
    df = load_data(os.path.join("..", "data", "processed", "bmt_dataset_processed.csv"))
    print("Data shape:", df.shape)
    
    # Ensure the target column exists.
    if "survival_status" not in df.columns:
        raise ValueError("Target column 'survival_status' not found in data.")
    
    # Separate features and target.
    X = df.drop(columns=["survival_status"])
    y = df["survival_status"]
    
    # Apply one-hot encoding to ensure all features are numeric (required for SMOTE).
    X = pd.get_dummies(X, drop_first=True)
    print("Feature shape after one-hot encoding:", X.shape)
    
    # Split data into training and testing sets (using stratification).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print("Training set shape:", X_train.shape, "; Test set shape:", X_test.shape)
    
    # -----------------------------
    # Train on original (non-oversampled) data.
    model_original = RandomForestClassifier(random_state=42)
    print("\nTraining on original data...")
    model_original.fit(X_train, y_train)
    metrics_original = evaluate_model(model_original, X_test, y_test)
    print("Performance on original data:")
    print(metrics_original)
    
    # -----------------------------
    # Oversample training data using SMOTE.
    print("\nApplying SMOTE to training data...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print("After SMOTE, training set shape:", X_train_smote.shape)
    
    # Train a new model on the SMOTE-balanced training data.
    model_smote = RandomForestClassifier(random_state=42)
    print("Training on SMOTE oversampled data...")
    model_smote.fit(X_train_smote, y_train_smote)
    metrics_smote = evaluate_model(model_smote, X_test, y_test)
    print("Performance on SMOTE oversampled data:")
    print(metrics_smote)
    
    # -----------------------------
    # Compare results.
    results_df = pd.DataFrame([metrics_original, metrics_smote], index=["Original", "SMOTE"])
    print("\nComparison of model performance:")
    print(results_df)
    
    # Optionally, plot a bar chart of ROC-AUC scores.
    results_df["ROC-AUC"].plot(kind="bar", title="ROC-AUC Comparison")
    plt.ylabel("ROC-AUC Score")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
