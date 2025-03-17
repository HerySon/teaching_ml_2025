import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore

def detect_outliers(df, method="iqr", contamination=0.05, z_threshold=3):
    """
    Détecte et marque les valeurs aberrantes en utilisant différentes méthodes.

    Paramètres :
        df: Le jeu de données.
        method : La méthode de détection ("iqr", "zscore", "isolation_forest").
        contamination (float) : La proportion attendue de valeurs aberrantes (pour Isolation Forest).
        z_threshold (float) : Seuil pour la méthode du Z-score.

    Retourne :
        pd.DataFrame : Le jeu de données avec une colonne 'outlier_tag' (1 = valeur aberrante, 0 = normal).
    """
    numerical_cols = df.select_dtypes(include=['number']).columns
    df_outliers = df.copy()

    if method == "iqr":
        # Détection avec la méthode IQR de Tukey
        Q1 = df[numerical_cols].quantile(0.25)
        Q3 = df[numerical_cols].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df_outliers["outlier_tag"] = ((df[numerical_cols] < lower_bound) | (df[numerical_cols] > upper_bound)).any(axis=1).astype(int)

    elif method == "zscore":
        # Détection avec la méthode Z-score
        z_scores = np.abs(df[numerical_cols].apply(zscore))
        df_outliers["outlier_tag"] = (z_scores > z_threshold).any(axis=1).astype(int)

    elif method == "isolation_forest":
        # Détection avec Isolation Forest
        model = IsolationForest(contamination=contamination, random_state=42)
        df_outliers["outlier_tag"] = model.fit_predict(df[numerical_cols])
        df_outliers["outlier_tag"] = df_outliers["outlier_tag"].apply(lambda x: 1 if x == -1 else 0)

    else:
        raise ValueError("Méthode invalide. Choisissez 'iqr', 'zscore' ou 'isolation_forest'.")

    return df_outliers

def handle_outliers(df, strategy="keep"):
    """
    Gère les valeurs aberrantes en fonction de la stratégie choisie.

    Paramètres :
        df (pd.DataFrame) : Jeu de données contenant une colonne 'outlier_tag'.
        strategy (str) : Stratégie de traitement ("keep", "remove", "median", "mean").

    Retourne :
        pd.DataFrame : Le jeu de données nettoyé.
    """
    if "outlier_tag" not in df.columns:
        raise ValueError("Le jeu de données ne contient pas de colonne 'outlier_tag'. Exécutez detect_outliers() d'abord.")

    df_cleaned = df.copy()

    if strategy == "keep":
        return df_cleaned.drop(columns=["outlier_tag"])

    elif strategy == "remove":
        return df_cleaned[df_cleaned["outlier_tag"] == 0].drop(columns=["outlier_tag"])

    elif strategy == "median":
        for col in df.select_dtypes(include=['number']).columns:
            median_val = df[col].median()
            df_cleaned.loc[df_cleaned["outlier_tag"] == 1, col] = median_val
        return df_cleaned.drop(columns=["outlier_tag"])

    elif strategy == "mean":
        for col in df.select_dtypes(include=['number']).columns:
            mean_val = df[col].mean()
            df_cleaned.loc[df_cleaned["outlier_tag"] == 1, col] = mean_val
        return df_cleaned.drop(columns=["outlier_tag"])

    else:
        raise ValueError("Stratégie invalide. Choisissez 'keep', 'remove', 'median' ou 'mean'.")

def plot_outliers(df, column, method="iqr", contamination=0.05, z_threshold=3):
    """
    Affiche un graphique pour visualiser les valeurs aberrantes d'une colonne.

    Paramètres :
        df (pd.DataFrame) : Le jeu de données.
        column (str) : La colonne à analyser.
        method (str) : La méthode de détection ("iqr", "zscore", "isolation_forest").
        contamination (float) : Proportion attendue d'outliers (pour Isolation Forest).
        z_threshold (float) : Seuil pour le Z-score.

    Retourne :
        None : Affiche directement le graphique.
    """
    if column not in df.columns:
        raise ValueError(f"La colonne '{column}' n'existe pas dans le DataFrame.")

    # Détection des valeurs aberrantes
    df_outliers = detect_outliers(df[[column]], method, contamination, z_threshold)

    # Séparer les outliers et les valeurs normales
    outliers = df_outliers[df_outliers["outlier_tag"] == 1]
    normal_values = df_outliers[df_outliers["outlier_tag"] == 0]

    # Création des figures
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot
    sns.boxplot(x=df_outliers[column], ax=axes[0], color="skyblue")
    axes[0].set_title(f"Boxplot de {column} ({method})")

    # Scatter plot
    sns.scatterplot(x=df_outliers.index, y=normal_values[column], color="blue", label="Normal", ax=axes[1])
    sns.scatterplot(x=outliers.index, y=outliers[column], color="red", label="Outliers", ax=axes[1])
    axes[1].set_title(f"Visualisation des outliers sur {column} ({method})")

    plt.show()
