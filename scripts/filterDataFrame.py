import pandas as pd
import numpy as np

def select_columns(df, category_threshold=50, ordinal_cols=None, downcast_level=None):
    """
    Selects relevant columns from the DataFrame by identifying numeric, ordinal, and non-ordinal categorical columns, 
    downcasting numeric types based on user preference, and filtering categorical columns by their number of unique categories.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame from which columns are selected.
    category_threshold : int, optional
        The maximum number of unique values allowed for non-ordinal categorical columns 
        to be included in the final selection (default is 50).
    ordinal_cols : list, optional
        A list of column names that are considered ordinal variables (default is None).
    downcast_level : dict, optional
        A dictionary specifying the level of downcasting for 'integer' and 'float' columns.
        Example: {"integer": "int32", "float": "float16"} (default is None, which applies int32/float32).

    Returns:
    -------
    tuple
        A tuple containing:
        - numeric_cols (pd.Index): The columns identified as numeric, possibly downcasted.
        - ordinal_cols (list): The columns identified as ordinal (user-defined or default empty list).
        - filtered_categorical_cols (list): The filtered non-ordinal categorical columns.

    Notes:
    ------
    - Numeric columns are downcasted based on user-specified levels (default is `int32` and `float32`).
    - Filtering of categorical columns is based on the number of unique categories.
    """
    if ordinal_cols is None:
        ordinal_cols = []  # Default to an empty list if not provided

    if downcast_level is None:
        downcast_level = {"integer": "int32", "float": "float32"}  # Default downcast levels

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    non_ordinal_cols = [col for col in categorical_cols if col not in ordinal_cols]
    
    # Downcasting numeric variables based on user-defined levels
    for col in numeric_cols:
        if df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast=downcast_level["integer"])
        elif df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast=downcast_level["float"])
    
    # Filtering categorical variables based on the number of unique values
    filtered_categorical_cols = [col for col in non_ordinal_cols if df[col].nunique() <= category_threshold]
    
    # Summary of selected columns
    print("Numeric columns:", numeric_cols)
    print("Ordinal columns:", ordinal_cols)
    print("Filtered non-ordinal categorical columns:", filtered_categorical_cols)
    
    # Return selected columns
    return numeric_cols, ordinal_cols, filtered_categorical_cols
