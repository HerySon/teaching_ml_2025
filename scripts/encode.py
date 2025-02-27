import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from category_encoders import HashingEncoder, CountEncoder

def delete_data(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    """Removes columns specified via columns_to_drop.

    Args:
        df (pd.DataFrame): DataFrame to clean.
        columns_to_drop (list): List of column names to remove.

    Returns:
        pd.DataFrame: DataFrame with specified columns removed.
    """
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df.drop(columns=columns_to_drop, axis=1, inplace=True)
    return df

def impute_numerical_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing values in numerical columns using KNN imputation.

    Args:
        df (pd.DataFrame): DataFrame containing numerical data.

    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    numeric_features = df.select_dtypes(include=['float', 'int'])
    imputer = KNNImputer(missing_values=np.nan)
    df[numeric_features.columns] = np.round(imputer.fit_transform(numeric_features), 1)
    return df

def impute_categorical_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing values in categorical columns using the most frequent value.

    Args:
        df (pd.DataFrame): DataFrame containing categorical data.

    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    categorical_features = df.select_dtypes(include=['object', 'category'])
    imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_features.columns] = imputer.fit_transform(categorical_features)
    return df

def filter_rare_categories(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """Groups rare categories (below a frequency threshold) into a single 'Other' category.

    Args:
        df (pd.DataFrame): DataFrame containing categorical features.
        threshold (float, optional): Minimum proportion of occurrences to keep a category. Defaults to 0.01.

    Returns:
        pd.DataFrame: DataFrame with rare categories grouped under 'Other'.
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        counts = df[col].value_counts(normalize=True)
        rare_categories = counts[counts < threshold].index
        df[col] = df[col].apply(lambda x: "Other" if x in rare_categories else x)
    return df

def encode_features_ohe(df: pd.DataFrame, output_dir: str = "encoded_features") -> None:
    """Encodes categorical features using OneHotEncoder and saves the output as a sparse matrix.

    Args:
        df (pd.DataFrame): DataFrame containing categorical features.
        output_dir (str, optional): Directory where encoded features will be saved. Defaults to "encoded_features".
    """
    os.makedirs(output_dir, exist_ok=True)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    for col in categorical_cols:

        encoded_feature = ohe.fit_transform(df[[col]])
        sp.save_npz(os.path.join(output_dir, f"{col}.npz"), encoded_feature)
        

def encode_features_hashing(df: pd.DataFrame, n_components: int = 8) -> pd.DataFrame:
    """Encodes categorical features using HashingEncoder.

    Args:
        df (pd.DataFrame): DataFrame containing categorical features.
        n_components (int, optional): Number of hash components. Defaults to 8.

    Returns:
        pd.DataFrame: DataFrame with hashed categorical features.
    """
    encoder = HashingEncoder(n_components=n_components)
    return encoder.fit_transform(df.select_dtypes(include=['object', 'category']))

def encode_features_count(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes categorical features using CountEncoder.

    Args:
        df (pd.DataFrame): DataFrame containing categorical features.

    Returns:
        pd.DataFrame: DataFrame with count-encoded categorical features.
    """
    encoder = CountEncoder()
    return encoder.fit_transform(df.select_dtypes(include=['object', 'category']))
