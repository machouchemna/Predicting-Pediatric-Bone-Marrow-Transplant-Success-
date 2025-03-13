import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set SHAP to display full feature names
shap.initjs()

def load_data(relative_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame using a path relative to this script.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, relative_path)
    print("Loading test data from:", full_path)
    df = pd.read_csv(full_path)
    print(f"Loaded test data with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def reindex_features(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    Ensure the feature set matches the trained model.
    """
    if hasattr(model, "feature_names_in_"):
        training_features = list(model.feature_names_in_)
        print(f"Reindexing test data to {len(training_features)} features...")
        X = X.reindex(columns=training_features, fill_value=0)
    else:
        print("Model does not have 'feature_names_in_'; using X as is.")
    return X

def load_latest_models(models_dir: str) -> dict:
    """
    Load the latest trained models dynamically from the models directory.
    """
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
    loaded_models = {}

    for model_name in ["RandomForest", "XGBoost", "LightGBM"]:
        latest_model_file = sorted(
            [f for f in model_files if f.startswith(model_name)], reverse=True
        )
        if latest_model_file:
            model_path = os.path.join(models_dir, latest_model_file[0])
            loaded_models[model_name] = joblib.load(model_path)
            print(f"✅ Loaded latest {model_name} model from {model_path}")
        else:
            print(f"⚠️ Warning: No trained {model_name} model found!")
    
    if not loaded_models:
        print("❌ No models were loaded. Please verify your models folder.")
    
    return loaded_models

def create_shap_plots(model, X_test, model_name, output_dir):
    """
    Generate SHAP summary plots for a given model.
    """
    print(f"Generating SHAP explanations for {model_name}...")

    try:
        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test)

        # SHAP Summary Plot
        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        summary_path = os.path.join(output_dir, f"{model_name}_shap_summary.png")
        plt.savefig(summary_path, bbox_inches="tight")
        print(f"📊 SHAP summary plot saved for {model_name}")

        # SHAP Feature Importance Plot
        plt.figure()
        shap.plots.bar(shap_values, show=False)
        bar_path = os.path.join(output_dir, f"{model_name}_shap_bar.png")
        plt.savefig(bar_path, bbox_inches="tight")
        print(f"📊 SHAP bar plot saved for {model_name}")

    except Exception as e:
        print(f"⚠️ SHAP analysis failed for {model_name}: {e}")

def main():
    # Load test data
    test_data_path = os.path.join("..", "data", "processed", "bmt_test.csv")
    df_test = load_data(test_data_path)
    
    # Check target column exists
    if "survival_status" not in df_test.columns:
        raise ValueError("Target column 'survival_status' not found in test data.")
    
    # Prepare features
    X_test = df_test.drop(columns=["survival_status"])
    X_test = pd.get_dummies(X_test, drop_first=True)  # Ensure numeric format
    print(f"Test data after encoding: {X_test.shape}")

    # Load latest models
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "..", "models")
    models = load_latest_models(models_dir)

    if not models:
        return  # Exit if no models were found

    # Create SHAP output directory
    shap_dir = os.path.join(script_dir, "..", "shap_analysis")
    os.makedirs(shap_dir, exist_ok=True)

    # Generate SHAP analysis for each model
    for model_name, model in models.items():
        print(f"\n🔍 Analyzing {model_name} with SHAP...")
        X_test_reindexed = reindex_features(X_test, model)
        create_shap_plots(model, X_test_reindexed, model_name, shap_dir)

if __name__ == "__main__":
    main()
