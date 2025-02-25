from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

def standard_scaler(df, numeric_cols):
    """
    Scale numeric columns using StandardScaler.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_cols (list): List of numeric column names.

    Returns:
    pd.DataFrame: DataFrame with numeric columns scaled.
    """
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def min_max_scaler(df, numeric_cols):
    """
    Scale numeric columns using MinMaxScaler.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_cols (list): List of numeric column names.

    Returns:
    pd.DataFrame: DataFrame with numeric columns scaled.
    """
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def robust_scaler(df, numeric_cols):
    """
    Scale numeric columns using RobustScaler.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_cols (list): List of numeric column names.

    Returns:
    pd.DataFrame: DataFrame with numeric columns scaled.
    """
    scaler = RobustScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def max_abs_scaler(df, numeric_cols):
    """
    Scale numeric columns using MaxAbsScaler.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_cols (list): List of numeric column names.

    Returns:
    pd.DataFrame: DataFrame with numeric columns scaled.
    """
    scaler = MaxAbsScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df
