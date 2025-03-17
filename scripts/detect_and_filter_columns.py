import pandas as pd
import numpy as np

def detect_and_filter_columns(df, max_categories=10):

    """
    Analyzes and classifies DataFrame columns into three types :
    - Numeric
    - Ordinal categorical
    - Non-ordinal categorical

    Parameters:
    df (pd.DataFrame): The input DataFrame to analyze.
    max_categories (int): The maximum number of unique values for a column to be considered ordinal.

    Returns:
    tuple: Three dictionaries containing the classified columns:
           - numeric_cols: Dictionary of numeric columns.
           - ordinal_cols: Dictionary of ordinal categorical columns.
           - non_ordinal_cols: Dictionary of non-ordinal categorical columns.
    """

    # dictionnaries to store columns types
    numeric_cols = {}
    ordinal_cols = {}
    non_ordinal_cols = {}

    types = pd.api.types

    for col in df.columns:
        if types.is_numeric_dtype(df[col]):
            # Downcasting des colonnes numériques
            if types.is_integer_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast='integer')
            else:
                df[col] = pd.to_numeric(df[col], downcast='float')
            numeric_cols[col] = df[col]

        elif types.is_categorical_dtype(df[col]) or df[col].dtype == "object":
            num_unique_values = df[col].nunique()
            if num_unique_values <= max_categories:

                # Ordinal if <= max_categories
                ordinal_cols[col] = df[col]
            else:
                non_ordinal_cols[col] = df[col]

    return numeric_cols, ordinal_cols, non_ordinal_cols