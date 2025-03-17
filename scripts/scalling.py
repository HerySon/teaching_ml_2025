# scaling.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

def standardization(df):
    """
    Scale the data using Standardization (Z-score normalization).

    Args:
        df (pandas.DataFrame): DataFrame containing the data to be scaled

    Returns:
        pandas.DataFrame: DataFrame with standardized data
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns)

def min_max_scaling(df):
    """
    Scale the data using Min-Max scaling.

    Args:
        df (pandas.DataFrame): DataFrame containing the data to be scaled

    Returns:
        pandas.DataFrame: DataFrame with data scaled between 0 and 1
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns)

def robust_scaling(df):
    """
    Scale the data using Robust Scaling (based on percentiles).

    Args:
        df (pandas.DataFrame): DataFrame containing the data to be scaled

    Returns:
        pandas.DataFrame: DataFrame with robust scaled data
    """
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns)

def max_abs_scaling(df):
    """
    Scale the data using MaxAbs Scaling (min-max scaling without outliers).

    Args:
        df (pandas.DataFrame): DataFrame containing the data to be scaled

    Returns:
        pandas.DataFrame: DataFrame with data scaled between -1 and 1
    """
    scaler = MaxAbsScaler()
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns)

def verify_scaling(df, feature):
    """
    Verify the scaling of a specific feature.

    Args:
        df (pandas.DataFrame): DataFrame containing the data
        feature (str): The feature to verify scaling for

    Returns:
        None
    """
    if feature not in df.columns:
        print(f"Feature '{feature}' not found in numeric columns.")
        return

    scaling_methods = {
        "Standardized": standardization,
        "Min-Max Scaled": min_max_scaling,
        "Robust Scaled": robust_scaling,
        "MaxAbs Scaled": max_abs_scaling
    }

    print(f"Statistics for '{feature}':")
    print(f"Original: Mean={df[feature].mean():.4f}, Std={df[feature].std():.4f}, Min={df[feature].min():.4f}, Max={df[feature].max():.4f}")

    for method_name, method in scaling_methods.items():
        scaled_data = method(df[[feature]])
        print(f"{method_name}: Mean={scaled_data[feature].mean():.4f}, Std={scaled_data[feature].std():.4f}, Min={scaled_data[feature].min():.4f}, Max={scaled_data[feature].max():.4f}")
