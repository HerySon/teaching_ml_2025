import pandas as pd
import numpy as np
from category_encoders import HashingEncoder, OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

def encode_categorical_features(df, ordinal_cols=None, threshold=10):
    """
    Encode categorical features using:
    - One-Hot Encoding for low-cardinality variables (<= low_threshold unique values)
    - Hashing Encoding for high-cardinality variables (>= high_threshold unique values)
    - Ordinal Encoding for manually specified ordinal variables

    Parameters:
    df (pd.DataFrame): Input dataframe
    ordinal_cols (list, optional): List of ordinal categorical columns to encode separately
    threshold (int, optional): Threshold to distinct low_cardinality and high-cardinality categorical columns

    Returns:
    pd.DataFrame: Encoded dataframe
    """

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Separate ordinal columns if provided
    ordinal_cols = ordinal_cols if ordinal_cols else []
    categorical_cols = [col for col in categorical_cols if col not in ordinal_cols]

    # Determine column cardinality
    unique_counts = df[categorical_cols].nunique()

    low_cardinality_cols = unique_counts[unique_counts <= threshold].index.tolist()
    high_cardinality_cols = unique_counts[unique_counts >= threshold].index.tolist()

    df_encoded = df.copy()

    # One-Hot Encoding for low-cardinality columns
    if low_cardinality_cols:
        ohe = OneHotEncoder(cols=low_cardinality_cols, use_cat_names=True)
        df_encoded = ohe.fit_transform(df_encoded)

    # Hashing Encoding for high-cardinality columns
    if high_cardinality_cols:
        for col in high_cardinality_cols:
            hashing_encoder = HashingEncoder(cols=col, n_components=16)  # Reduce to 16 features
            transformed = hashing_encoder.fit_transform(df_encoded[[col]])
            transformed.columns = [f"{col}_hash_{i}" for i in range(16)]  # Rename
            df_encoded = pd.concat([df_encoded, transformed], axis=1)  # Add to final dataframe
            df_encoded.drop(columns=[col], inplace=True)

    # Ordinal Encoding for manually defined ordinal columns
    if ordinal_cols:
        ord_enc = OrdinalEncoder()
        df_encoded[ordinal_cols] = ord_enc.fit_transform(df_encoded[ordinal_cols])

    print(f"Categorical Columns Identified:")
    print(f"- Low cardinality (One-Hot): {low_cardinality_cols}")
    print(f"- High cardinality (Hashing): {high_cardinality_cols}")
    print(f"- Ordinal Columns (Ordinal Encoding): {ordinal_cols}")

    return df_encoded