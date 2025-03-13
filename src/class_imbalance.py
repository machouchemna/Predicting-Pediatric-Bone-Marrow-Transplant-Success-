import os
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    raise ImportError("Please install imbalanced-learn (pip install imbalanced-learn) to use SMOTE.")

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight


def load_data(relative_path: str) -> pd.DataFrame:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, relative_path)
    print("Loading data from:", full_path)
    return pd.read_csv(full_path)


def create_plots_dir():
    """
    Create a directory for saving imbalance-related plots.
    Ensure the directory is successfully created and return the absolute path.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.abspath(os.path.join(script_dir, "..", "imbalance_plots"))

    try:
        os.makedirs(plots_dir, exist_ok=True)
        if os.path.exists(plots_dir):
            print(f"✅ Imbalance plots directory successfully created: {plots_dir}")
        else:
            raise FileNotFoundError(f"⚠️ Failed to create the directory: {plots_dir}")
    except Exception as e:
        print(f"❌ Error creating the imbalance_plots directory: {e}")
    
    return plots_dir



def plot_class_distribution(y: pd.Series, title: str, filename: str, folder: str):
    """
    Plot and save the class distribution as a bar chart.
    """
    if not os.path.exists(folder):
        print(f"⚠️ WARNING: The folder {folder} does not exist! Creating it now...")
        os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.countplot(x=y)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    
    filepath = os.path.join(folder, filename)
    
    try:
        plt.savefig(filepath, bbox_inches="tight")
        print(f"✅ Plot successfully saved to: {filepath}")
    except Exception as e:
        print(f"❌ Error saving plot: {e}")

    plt.close()



def balance_with_smote(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    print("Applying SMOTE for oversampling...")
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res


def compute_class_weights(y: pd.Series) -> dict:
    classes = np.unique(y)
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y)
    return dict(zip(classes, weights))


def split_and_save_train_test(X, y, test_size: float = 0.2, random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=random_state, stratify=y)

    # Apply SMOTE only on training data
    imbalance_ratio = max(Counter(y_train).values()) / min(Counter(y_train).values())
    if imbalance_ratio > 1.5:  # Only apply if class imbalance is significant
        X_train, y_train = balance_with_smote(X_train, y_train, random_state=random_state)
        print("Applied SMOTE to the training set.")

    train_df = X_train.copy()
    train_df["survival_status"] = y_train
    test_df = X_test.copy()
    test_df["survival_status"] = y_test

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, "..", "data", "processed", "bmt_train.csv")
    test_path = os.path.join(script_dir, "..", "data", "processed", "bmt_test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Training data saved to:", train_path)
    print("Testing data saved to:", test_path)


def main():
    # Load processed data.
    input_csv = os.path.join("..", "data", "processed", "bmt_dataset_processed.csv")
    df = load_data(input_csv)
    print("Original processed data shape:", df.shape)

    if "survival_status" not in df.columns:
        raise ValueError("Target column 'survival_status' not found in data.")

    # Separate features and target.
    X = df.drop(columns=["survival_status"])
    y = df["survival_status"]

    # Ensure all features are numeric before applying SMOTE.
    if not np.issubdtype(X.dtypes, np.number):
        X = pd.get_dummies(X, drop_first=True)
    print("After one-hot encoding, feature shape:", X.shape)

    # Create directory for saving plots.
    plots_dir = create_plots_dir()

    # Plot and save the original class distribution.
    print("Original class distribution:", Counter(y))
    plot_class_distribution(y, title="Original Class Distribution", 
                            filename="original_class_distribution.png", folder=plots_dir)

    # Apply SMOTE for oversampling
    print("Applying SMOTE for oversampling...")
    X_smote, y_smote = balance_with_smote(X, y, random_state=42)
    
    print("After SMOTE, class distribution:", Counter(y_smote))

    # 🔹 Ensure the SMOTE plot is generated and saved
    plot_class_distribution(y_smote, title="SMOTE Oversampled Distribution", 
                            filename="smote_distribution.png", folder=plots_dir)

    # Compute and print class weights (useful for training)
    smote_class_weights = compute_class_weights(y_smote)
    print("Computed class weights after SMOTE:", smote_class_weights)

    # Split and save the oversampled data into training and testing sets
    split_and_save_train_test(X_smote, y_smote, test_size=0.2, random_state=42)

if __name__ == "__main__":
    main()
