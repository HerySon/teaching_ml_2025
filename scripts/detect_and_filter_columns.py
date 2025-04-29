import pandas as pd

def detect_and_filter_columns(df, max_categories=50, downcast=True, max_unique_for_ordinal=None):
    """
    Identifie et sépare les colonnes numériques, ordinales et non ordinales d'un DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame à analyser.
        max_categories (int, optionnel): Le nombre maximum de catégories pour qu'une colonne soit considérée comme ordinale (par défaut 50).
        downcast (bool, optionnel): Si True, les colonnes numériques seront optimisées pour la mémoire (par défaut True).
        max_unique_for_ordinal (int, optionnel): Le nombre maximal de valeurs uniques pour identifier les colonnes ordinales. Si None, utilise `max_categories`.

    Returns:
        tuple: Un tuple avec trois dictionnaires :
            - `numeric_cols`: Les colonnes numériques 
            - `ordinal_cols`: Les colonnes ordinales 
            - `non_ordinal_cols`: Les colonnes non ordinales.
"""
    numeric_cols = {}
    ordinal_cols = {}
    non_ordinal_cols = {}
    verif_type = pd.api.types

    if max_unique_for_ordinal is None:
        max_unique_for_ordinal = max_categories

    for col in df.columns:
        if verif_type.is_numeric_dtype(df[col]):
            # Downcasting des colonnes numériques
            if downcast:
                if verif_type.is_integer_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                else:
                    df[col] = pd.to_numeric(df[col], downcast='float')
            numeric_cols[col] = df[col]

        elif verif_type.is_categorical_dtype(df[col]) or df[col].dtype == "object":
            num_unique_values = df[col].nunique()
            if num_unique_values <= max_unique_for_ordinal:
                ordinal_cols[col] = df[col]
            else:
                non_ordinal_cols[col] = df[col]

    return numeric_cols, ordinal_cols, non_ordinal_cols

print("Fonction importée avec succès !")
