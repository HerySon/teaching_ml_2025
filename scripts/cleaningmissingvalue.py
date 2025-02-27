import re
from sklearn.impute import SimpleImputer, KNNImputer
import pandas as pd
import numpy as np

def detect_Cleaning_data(df, extra_columns_to_remove=None, suffixes_to_drop=None):
    """
    Cleans the DataFrame by extracting numeric values from `serving_size`, optionally removing specific columns 
    or those with certain suffixes, and keeping only the '_tags' variant when multiple versions exist.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be cleaned.
    extra_columns_to_remove (list, optional): Specific columns to remove. If None, no columns are removed.
    suffixes_to_drop (list, optional): List of suffixes for columns to be dropped. If None, no suffix-based columns are removed.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """

    # Delete duplicated lines and null lines
    df.drop_duplicates(inplace=True)
    df.dropna(how='all', inplace=True)

    # Step 1: Clean the serving_size column
    df["serving_size"] = df["serving_size"].str.extract(r'([\d\.]+)', expand=False)
    df["serving_size"] = pd.to_numeric(df["serving_size"], errors='coerce')

    # Step 2: Drop columns based on suffixes (only if suffixes_to_drop is provided)
    columns_to_drop = []
    if suffixes_to_drop:
        columns_to_drop.extend([col for col in df.columns if any(col.endswith(suffix) for suffix in suffixes_to_drop)])

    # Step 3: Drop specific columns (only if extra_columns_to_remove is provided)
    if extra_columns_to_remove:
        columns_to_drop.extend([col for col in extra_columns_to_remove if col in df.columns])

    # Step 4: Keep only '_tags' variant if multiple column versions exist
    grouped_columns = {}

    for col in df.columns:
        base_name = col.split('_')[0]  # Extract base name before suffix
        if base_name not in grouped_columns:
            grouped_columns[base_name] = []
        grouped_columns[base_name].append(col)

    for base_name, cols in grouped_columns.items():
        if len(cols) > 1:
            to_keep = next((col for col in cols if col.endswith('_tags')), None)
            if to_keep:
                to_remove = [col for col in cols if col != to_keep]
                columns_to_drop.extend(to_remove)

    # Step 5: Drop the identified columns only if there are any to drop
    columns_to_drop = list(set(columns_to_drop))  # Remove duplicates

    if columns_to_drop:  
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore', inplace=True)
        print("Dropped columns:", columns_to_drop)
    else:
        print("No columns were dropped.")

    return df  # Return the cleaned DataFrame


def cleaning_Missing_Data(df, threshold):
    """
    Removes columns with a percentage of missing data above a given threshold.

    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame to clean.
    threshold : float
        The percentage threshold above which columns with missing data are removed.

    Returns:
    -------
    None
        The function modifies the DataFrame in place and prints the number of columns removed.
    """
    percent_missing = df.isnull().sum() * 100 / len(df)
    # Select columns with a percentage of missing values higher than the threshold
    columns_to_drop = percent_missing[percent_missing.values > threshold].index
    # Drop columns selected, in the df with inplace=True
    df.drop(columns_to_drop, axis='columns', inplace=True)
    return print(f'Retrait de {len(columns_to_drop)} colonnes de votre dataframe')



def impute_missing_values(df):    
    """
    Impute missing values in the given DataFrame.
    
    Numeric columns are imputed using K-Nearest Neighbors (KNN), 
    while non-numeric columns are imputed using the most frequent value.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame with potential missing values.
    
    Returns:
    None: The function modifies the DataFrame in place.
    """
    
    # Identify numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns
    non_numeric_cols = df.select_dtypes(exclude=['float', 'int']).columns

    # Impute non-numeric columns using the most frequent value
    if not non_numeric_cols.empty:
        imputer_categorical = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        df[non_numeric_cols] = imputer_categorical.fit_transform(df[non_numeric_cols])
    
    # Impute numeric columns using KNNImputer
    if not numeric_cols.empty:
        imputer_numeric = KNNImputer(missing_values=np.nan)
        df[numeric_cols] = imputer_numeric.fit_transform(df[numeric_cols])


