import pandas as pd
import numpy as np

def select_columns(df, category_threshold=50):
    """
    Selects relevant columns from the DataFrame by identifying numeric, ordinal, and non-ordinal categorical columns, 
    downcasting numeric types, and filtering categorical columns by their number of unique categories.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame from which columns are selected.
    category_threshold : int, optional
        The maximum number of unique values allowed for non-ordinal categorical columns 
        to be included in the final selection (default is 50).

    Returns:
    -------
    tuple
        A tuple containing:
        - numeric_cols (pd.Index): The columns identified as numeric, possibly downcasted.
        - ordinal_cols (list): The columns identified as ordinal (currently empty).
        - filtered_categorical_cols (list): The filtered non-ordinal categorical columns.

    Notes:
    ------
    - Numeric columns are downcasted to more memory-efficient types (`int32`, `float32`) when possible.
    - The current implementation assumes the `ordinal_cols` list is defined manually, which may need adjustment.
    - Filtering of categorical columns is based on the number of unique categories.
    """
    # Identifications des types de colonnes
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    ordinal_cols = []  # Liste de variables ordinales (à définir selon le contexte)
    non_ordinal_cols = [col for col in categorical_cols if col not in ordinal_cols]
    
    # Downcasting des variables numériques
    for col in numeric_cols:
        if df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Filtrage des variables catégorielles par le nombre de catégories
    filtered_categorical_cols = [col for col in non_ordinal_cols if df[col].nunique() <= category_threshold]
    
    # Résumé des colonnes sélectionnées
    print("Colonnes numériques:", numeric_cols)
    print("Colonnes ordinales:", ordinal_cols)
    print("Colonnes catégorielles non ordinales filtrées:", filtered_categorical_cols)
    
    # Retourner les colonnes à conserver
    return numeric_cols, ordinal_cols, filtered_categorical_cols
