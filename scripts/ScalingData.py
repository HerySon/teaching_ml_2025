from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd

def scale_features(df, method="auto"):
    """
    Scales numerical features in the dataframe.

    Parameters:
    df (pd.DataFrame): The dataframe containing numerical features.
    method (str): Scaling method - "standard" for StandardScaler, "minmax" for MinMaxScaler,
                  "auto" to decide based on skewness.

    Returns:
    pd.DataFrame: Scaled dataframe.
    """

    df_scaled = df.copy()

    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns

    if method == "auto":
        # Mean skewness of the DataFrame
        mean_skewness = abs(df[numerical_cols].skew()).mean()
        
        # if mean skewness > 1, apply MinMaxScaler, else StandardScaler
        if mean_skewness > 1:
            scaler = MinMaxScaler()
            print("Applying MinMaxScaler to the entire DataFrame (skewed data).")
        else:
            scaler = StandardScaler()
            print("Applying StandardScaler to the entire DataFrame (normal-like data).")
        
        df_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_cols]), columns=numerical_cols)

    elif method == "standard":
        scaler = StandardScaler()
        df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])

    elif method == "minmax":
        scaler = MinMaxScaler()
        df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])

    else:
        raise ValueError("Invalid method. Choose 'auto', 'standard', or 'minmax'.")

    return df_scaled
