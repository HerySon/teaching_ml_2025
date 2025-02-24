import pandas as pd
import numpy as np
from collections import Counter

def remove_duplicates(df, subset='code'):
    """
    Remove duplicate rows from the DataFrame based on a subset of columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    subset (str or list): Column(s) to consider for identifying duplicates.

    Returns:
    pd.DataFrame: DataFrame with duplicates removed.
    """
    df = df.drop_duplicates(subset=subset)
    return df

import pandas as pd

def remove_unnecessary_columns(df):
    """
    Removes unnecessary columns from the given DataFrame if they exist.
    
    :param df: pandas DataFrame
    :return: Cleaned DataFrame
    """
    columns_to_remove = [
        "url", "creator", "created_t", "created_datetime", 
        "last_modified_t", "last_modified_datetime", 
        "last_modified_by", "last_updated_t", "last_updated_datetime"
    ]
    
    df = df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors="ignore")
    
    return df


def handle_missing_values(df, numeric_cols, threshold=0.5):
    """
    Handle missing values in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_cols (list): List of numeric column names.
    threshold (float): Threshold for dropping columns with missing values.

    Returns:
    pd.DataFrame: DataFrame with missing values handled.
    """
    # Fill missing values in numeric columns with the mean
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Drop columns with more than threshold% missing values
    df = df.dropna(axis=1, thresh=int(threshold * len(df)))

    return df

def detect_outliers(df, n, features):
    """
    Detect outliers in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    n (int): Number of outliers to consider.
    features (list): List of feature column names to check for outliers.

    Returns:
    list: List of indices of outliers.
    """
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    return multiple_outliers

def remove_outliers(df, outliers):
    """
    Remove outliers from the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    outliers (list): List of indices of outliers to remove.

    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    df = df.drop(outliers, axis=0).reset_index(drop=True)
    return df

def knn_imputer(df, numeric_cols, n_neighbors=5):
    """
    Impute missing values in numeric columns using KNNImputer.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_cols (list): List of numeric column names.
    n_neighbors (int): Number of neighbors to use for KNN imputation.

    Returns:
    pd.DataFrame: DataFrame with missing values imputed.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df

def simple_imputer(df, numeric_cols, strategy='mean'):
    """
    Impute missing values in numeric columns using SimpleImputer.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_cols (list): List of numeric column names.
    strategy (str): Strategy to use for imputation ('mean', 'median', 'most_frequent', 'constant').

    Returns:
    pd.DataFrame: DataFrame with missing values imputed.
    """
    imputer = SimpleImputer(strategy=strategy)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df