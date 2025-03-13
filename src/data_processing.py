"""
data_processing.py

This module performs data preprocessing tasks:
1. Handle missing values and invalid characters.
2. Remove highly correlated features.
3. Optimize memory usage by downcasting numeric types.
4. Encode categorical features properly.
5. Handle outliers using the IQR method.
6. Save the processed data to a new CSV file.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and replace invalid characters like "?" with NaN.
    
    - Numeric columns: Fill with median.
    - Categorical columns: Fill with the most frequent value.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled.
    """
    df = df.copy()

    # Replace "?" and other placeholders with NaN
    df.replace(["?", "unknown", "N/A", ""], np.nan, inplace=True)

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    # Fill missing values
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df


def remove_highly_correlated_features(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    Remove features that are highly correlated (above a given threshold).
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    threshold : float, default 0.9
        Correlation threshold.

    Returns
    -------
    pd.DataFrame
        DataFrame with highly correlated features removed.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr().abs()

    # Upper triangle matrix (to avoid duplicate comparisons)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    df.drop(columns=to_drop, inplace=True, errors='ignore')
    return df


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize memory usage by downcasting numeric columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        Memory-optimized DataFrame.
    """
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='float')
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables using Label Encoding.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with categorical variables encoded.
    """
    df = df.copy()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    encoder = LabelEncoder()

    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col].astype(str))

    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle outliers using the Interquartile Range (IQR) method.
    
    Any value beyond 1.5 * IQR is capped at the upper or lower bound.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with outliers treated.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    return df


def save_processed_data(df: pd.DataFrame, relative_output_path: str) -> None:
    """
    Save the processed DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        The processed DataFrame.
    relative_output_path : str
        Relative path to save the new CSV file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, relative_output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Processed data saved to:", output_path)


def main():
    """
    Load the dataset, apply preprocessing, and save the cleaned data.
    """
    # Relative path to raw data
    input_csv = os.path.join("..", "data", "processed", "bmt_dataset.csv")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_input_path = os.path.join(script_dir, input_csv)
    print("Loading raw data from:", full_input_path)

    # Load the dataset
    df = pd.read_csv(full_input_path)
    print("Original data shape:", df.shape)

    # 1. Handle missing values and invalid entries
    df = handle_missing_values(df)
    print("After handling missing values, shape:", df.shape)

    # 2. Remove highly correlated features
    df = remove_highly_correlated_features(df, threshold=0.9)
    print("After removing highly correlated features, shape:", df.shape)

    # 3. Encode categorical variables
    df = encode_categorical_features(df)
    print("After encoding categorical variables, shape:", df.shape)

    # 4. Handle outliers
    df = handle_outliers(df)
    print("After handling outliers, shape:", df.shape)

    # 5. Optimize memory usage
    df = optimize_memory(df)
    print("Memory optimization complete.")

    # 6. Save the cleaned dataset
    output_csv = os.path.join("..", "data", "processed", "bmt_dataset_processed.csv")
    save_processed_data(df, output_csv)


if __name__ == "__main__":
    main()
