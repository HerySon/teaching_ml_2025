import pandas as pd
import numpy as np


def detect_outliers_iqr(df):
    """
    Détecte les valeurs aberrantes dans toutes les colonnes numériques du DataFrame en utilisant la méthode de l'IQR (Interquartile Range).
    Une valeur est considérée comme aberrante si elle est en dehors de l'intervalle [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR].

    Paramètres :
    df (pd.DataFrame) : Le DataFrame contenant les données.

    Retourne :
    pd.DataFrame : Un DataFrame contenant les valeurs aberrantes détectées.
    """
    outliers = pd.DataFrame()
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = pd.concat([outliers, df[(df[column] < lower_bound) | (df[column] > upper_bound)]])
    return outliers.drop_duplicates()


def impute_outliers(df, method='median'):
    """
    Remplace les valeurs aberrantes par la médiane ou la moyenne de la colonne.

    Paramètres :
    df (pd.DataFrame) : Le DataFrame contenant les données.
    method (str) : Méthode d'imputation ('median' pour la médiane, 'mean' pour la moyenne).

    Retourne :
    pd.DataFrame : Un DataFrame avec les valeurs aberrantes remplacées.
    """
    imputed_df = df.copy()
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        if method == 'median':
            replacement_value = df[column].median()
        elif method == 'mean':
            replacement_value = df[column].mean()
        else:
            raise ValueError("Méthode d'imputation non reconnue. Utilisez 'median' ou 'mean'.")

        imputed_df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = replacement_value

    return imputed_df


def remove_outliers(df):
    """
    Supprime les valeurs aberrantes en utilisant la méthode de l'IQR (Interquartile Range).

    Paramètres :
    df (pd.DataFrame) : Le DataFrame contenant les données.

    Retourne :
    pd.DataFrame : Un DataFrame nettoyé sans valeurs aberrantes.
    """
    clean_df = df.copy()
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        clean_df = clean_df[(clean_df[column] >= lower_bound) & (clean_df[column] <= upper_bound)]
    return clean_df