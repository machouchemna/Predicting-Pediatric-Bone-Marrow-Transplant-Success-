import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

def load_data(relative_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame using a path relative to this script.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, relative_path)
    print("Loading training data from:", full_path)
    return pd.read_csv(full_path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all categorical variables are one-hot encoded and remove highly correlated features.
    """
    # One-hot encode categorical variables
    df = pd.get_dummies(df, drop_first=True)
    
    # Remove highly correlated features (threshold > 0.9)
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
    
    if to_drop:
        print(f"Removing {len(to_drop)} highly correlated features: {to_drop}")
        df.drop(columns=to_drop, inplace=True)

    return df

def plot_feature_importance(model, X, model_name, output_dir, top_n=20):
    """
    Plot and save feature importance for tree-based models.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = X.columns
        sorted_idx = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[sorted_idx][:top_n], y=[feature_names[i] for i in sorted_idx[:top_n]])
        plt.title(f"Top {top_n} Feature Importances - {model_name}")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()

        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{model_name}_feature_importance.png"))
        plt.close()
        print(f"Feature importance plot saved for {model_name}")

def train_models(X_train, y_train, X_val, y_val):
    """
    Train multiple models and evaluate them on training and validation sets.
    """
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "LightGBM": LGBMClassifier(random_state=42, min_child_samples=10, verbose=-1)  # Prevents LightGBM warnings
    }

    trained_models = {}
    performance_results = {}

    for name, model in models.items():
        print(f"\nTraining {name} on {X_train.shape[1]} features...")
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Probabilities for ROC-AUC
        y_train_prob = model.predict_proba(X_train)[:, 1] if hasattr(model, "predict_proba") else None
        y_val_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

        # Metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)

        train_auc = roc_auc_score(y_train, y_train_prob) if y_train_prob is not None else 0.0
        val_auc = roc_auc_score(y_val, y_val_prob) if y_val_prob is not None else 0.0

        print(f"{name} Training Accuracy: {train_acc:.4f}, ROC-AUC: {train_auc:.4f}")
        print(f"{name} Validation Accuracy: {val_acc:.4f}, ROC-AUC: {val_auc:.4f}")
        print(f"Validation Classification Report:\n{classification_report(y_val, y_val_pred)}")

        # Store results
        performance_results[name] = {
            "Train Accuracy": train_acc, "Train ROC-AUC": train_auc,
            "Validation Accuracy": val_acc, "Validation ROC-AUC": val_auc
        }

        trained_models[name] = model

    return trained_models, performance_results

def save_models(models: dict, output_dir: str):
    """
    Save each trained model as a pickle file with a timestamp.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    for name, model in models.items():
        filename = os.path.join(output_dir, f"{name}_model_{timestamp}.pkl")
        joblib.dump(model, filename)
        print(f"✅ Saved {name} model to {filename}")

def save_performance(performance_results: dict, output_dir: str):
    """
    Save model performance metrics to CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    performance_df = pd.DataFrame(performance_results).T
    performance_path = os.path.join(output_dir, "validation_performance.csv")
    performance_df.to_csv(performance_path)
    print(f"📊 Validation performance saved to {performance_path}")

def main():
    # Load training data
    train_csv = os.path.join("..", "data", "processed", "bmt_train.csv")
    df = load_data(train_csv)
    print("Training data shape:", df.shape)

    # Ensure target column exists
    if "survival_status" not in df.columns:
        raise ValueError("❌ Target column 'survival_status' not found in training data.")

    # Preprocess data
    df = preprocess_data(df)

    # Separate features and target
    X = df.drop(columns=["survival_status"])
    y = df["survival_status"]

    # Split into train (80%) and validation (20%)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")

    # Train models
    trained_models, performance_results = train_models(X_train, y_train, X_val, y_val)

    # Save models with timestamps
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "..", "models")
    save_models(trained_models, models_dir)

    # Save performance metrics
    performance_dir = os.path.join(script_dir, "..", "model_performance")
    save_performance(performance_results, performance_dir)

    # Save feature importance plots
    plots_dir = os.path.join(script_dir, "..", "feature_importance_plots")
    os.makedirs(plots_dir, exist_ok=True)

    for name, model in trained_models.items():
        plot_feature_importance(model, X_train, name, plots_dir)

if __name__ == "__main__":
    main()
