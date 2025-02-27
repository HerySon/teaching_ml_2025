import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore

def detect_outliers(df, method="iqr", contamination=0.05):
    """
    Detect and tag outliers using different methods.
    
    Parameters:
        df (pd.DataFrame): The dataset.
        method (str): The detection method ("iqr", "zscore", "isolation_forest").
        contamination (float): The expected proportion of outliers (only for Isolation Forest).

    Returns:
        pd.DataFrame: The dataset with an 'outlier_tag' column (1 = outlier, 0 = normal).
    """
    df_outliers = df.copy()
    numerical_cols = df.select_dtypes(include=['number']).columns  # Select numerical columns
    
    if method == "iqr":
        # Outlier detection using the Tukey's IQR rule
        Q1 = df[numerical_cols].quantile(0.25)
        Q3 = df[numerical_cols].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_outliers["outlier_tag"] = ((df[numerical_cols] < lower_bound) | (df[numerical_cols] > upper_bound)).any(axis=1).astype(int)
        print("Outliers detected using the IQR method.")

    elif method == "zscore":
        # Outlier detection using Z-score
        threshold = 3
        z_scores = np.abs(df[numerical_cols].apply(zscore))
        df_outliers["outlier_tag"] = (z_scores > threshold).any(axis=1).astype(int)
        print("Outliers detected using the Z-score method.")

    elif method == "isolation_forest":
        # Outlier detection using Isolation Forest
        model = IsolationForest(contamination=contamination, random_state=42)
        df_outliers["outlier_tag"] = model.fit_predict(df[numerical_cols])
        df_outliers["outlier_tag"] = df_outliers["outlier_tag"].apply(lambda x: 1 if x == -1 else 0)
        print("Outliers detected using Isolation Forest.")

    else:
        raise ValueError("Invalid method. Choose 'iqr', 'zscore' or 'isolation_forest'.")

    return df_outliers

def handle_outliers(df, strategy="keep"):
    """
    Handle outliers based on the selected strategy.
    
    Parameters:
        df (pd.DataFrame): Dataset containing an 'outlier_tag' column.
        strategy (str): The handling strategy ("keep", "remove", "median", "mean").

    Returns:
        pd.DataFrame: The cleaned dataset.
    """
    if "outlier_tag" not in df.columns:
        raise ValueError("The dataset does not contain an 'outlier_tag' column. Run detect_outliers() first.")

    df_cleaned = df.copy()
    
    if strategy == "keep":
        print("Outliers are kept.")
        return df_cleaned.drop(columns=["outlier_tag"])

    elif strategy == "remove":
        print("Outliers are removed.")
        return df_cleaned[df_cleaned["outlier_tag"] == 0].drop(columns=["outlier_tag"])

    elif strategy == "median":
        print("Replacing outliers with the median.")
        for col in df.select_dtypes(include=['number']).columns:
            median_val = df[col].median()
            df_cleaned.loc[df_cleaned["outlier_tag"] == 1, col] = median_val
        return df_cleaned.drop(columns=["outlier_tag"])

    elif strategy == "mean":
        print("Replacing outliers with the mean.")
        for col in df.select_dtypes(include=['number']).columns:
            mean_val = df[col].mean()
            df_cleaned.loc[df_cleaned["outlier_tag"] == 1, col] = mean_val
        return df_cleaned.drop(columns=["outlier_tag"])

    else:
        raise ValueError("Invalid strategy. Choose 'keep', 'remove', 'median' or 'mean'.")

