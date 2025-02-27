import re
from sklearn.impute import SimpleImputer, KNNImputer
import pandas as pd
import numpy as np

def detectCleaningdata(df):
    """
    Clean the DataFrame by:
    - Clean the `serving_size` column by extracting only the numeric part and discarding units.
    - Dropping columns with suffixes '_t' or '_datetime'.
    - Removing specified columns like 'url' and 'creator' (if they exist).
    - Keeping only the '_tags' variant when multiple versions of the same column exist
      (such as 'categories', 'categories_tags', 'categories_en').

    Parameters:
    df (pd.DataFrame): The input DataFrame to be cleaned.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    #Step 1: Clean the serving_size
    # Use regex to extract numeric part (handles both integers and decimals)
    df["serving_size"] = df["serving_size"].str.extract(r'([\d\.]+)', expand=False)
    
    # Convert extracted numbers to float and handle NaN values
    df["serving_size"] = pd.to_numeric(df["serving_size"], errors='coerce')

    # Step 2: Drop columns based on suffix
    suffixes_to_drop = ['_t', '_datetime', '_url', '_by']
    columns_to_drop = [col for col in df.columns if any(col.endswith(suffix) for suffix in suffixes_to_drop)]

    # Manually add specific columns to drop
    columns_to_remove = ['url', 'creator']
    columns_to_drop.extend([col for col in columns_to_remove if col in df.columns])

    # Step 3: Keep only '_tags' variant if multiple column versions exist
    # Identify related columns and keep only '_tags' if present
    grouped_columns = {}

    for col in df.columns:
        # Extract base name by removing the suffix (_tags, _en, etc.) if it exists
        base_name = col.split('_')[0]  # Always take the first part
        if base_name not in grouped_columns:
            grouped_columns[base_name] = []
        grouped_columns[base_name].append(col)

    for base_name, cols in grouped_columns.items():
        if len(cols) > 1:
            # Check if a '_tags' column exists
            to_keep = next((col for col in cols if col.endswith('_tags')), None)
            if to_keep:
                # Remove all columns except the '_tags' column
                to_remove = [col for col in cols if col != to_keep]
                columns_to_drop.extend(to_remove)

    # Step 4: Drop columns from the DataFrame
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]

    # Drop the identified columns
    df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

    print("Dropped columns:", columns_to_drop)


def cleaningMissingData(df, threshold):
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


