import re
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

def drop_unnecessary_columns(df, cols_to_drop=None):
    """Supprime les colonnes non pertinentes
    
    Args:
        df (pd.DataFrame): Le DataFrame à traiter.
        cols_to_drop (list, optional): Liste des colonnes à supprimer. 
                                       Par défaut, les colonnes définies dans la fonction seront supprimées.
    
    Returns:
        pd.DataFrame: Le DataFrame nettoyé.
    """
    if cols_to_drop is None:
        cols_to_drop = [
            "code", "url", "creator",
            "last_modified_t", "last_modified_datetime",
            "image_url", "image_small_url","categories","categories_en"
        ]
    
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    return df

def drop_columns_with_too_many_missing(df, threshold):
    """Supprime les colonnes contenant plus de `threshold`% de valeurs manquantes, sauf 'serving_size'.
    
    Args:
        df (pd.DataFrame): Le DataFrame à traiter.
        threshold (int, optional): Pourcentage de valeurs manquantes maximal autorisé. 
                                   
    Returns:
        pd.DataFrame: Le DataFrame nettoyé.
    """
    percent_missing = df.isnull().mean() * 100
    columns_to_drop = percent_missing[percent_missing > threshold].index
    
    # Exclure 'serving_size' de la suppression
    columns_to_drop = [col for col in columns_to_drop if col != 'serving_size']
    
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    return df


def impute_missing_values(df, categorical_columns=None, numeric_imputer=None):
    """Impute les valeurs manquantes pour les colonnes catégorielles et numériques.
    
    Args:
        df (pd.DataFrame): Le DataFrame à traiter.
        categorical_columns (list, optional): Liste des colonnes catégorielles à imputer.
                                              Par défaut, une liste pré-définie sera utilisée.
        numeric_imputer (sklearn.impute, optional): Imputateur numérique à utiliser (ex: SimpleImputer, KNNImputer).
                                                   Par défaut, KNNImputer.
    
    Returns:
        pd.DataFrame: Le DataFrame avec valeurs imputées.
    """
    # Sélectionner les colonnes catégorielles (de type object ou string)
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()

    # Imputation des valeurs catégorielles (mode)
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = imputer_categorical.fit_transform(df[categorical_columns])

    # Sélectionner les colonnes numériques
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Imputation des valeurs numériques (moyenne ou KNN par défaut)
    if numeric_imputer is None:
        numeric_imputer = KNNImputer(missing_values=np.nan)
    
    df[numeric_features] = numeric_imputer.fit_transform(df[numeric_features])
    
    return df

def clean_serving_size(df):
    """Extrait les valeurs numériques de 'serving_size' et les convertit en float.
    
    Args:
        df (pd.DataFrame): Le DataFrame à traiter.
    
    Returns:
        pd.DataFrame: Le DataFrame avec 'serving_size' nettoyé.
    """
    def extract_numeric(value):
        # Recherche de la première séquence numérique dans la chaîne
        match = re.search(r"(\d+(\.\d+)?)", str(value))  # Capture les nombres entiers et à virgule
        return float(match.group(1)) if match else np.nan  # Retourne le nombre ou NaN si rien n'est trouvé

    # Appliquer la fonction d'extraction
    df["serving_size"] = df["serving_size"].apply(extract_numeric)
    
    return df
