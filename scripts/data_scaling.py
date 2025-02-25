import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

def standard_scaler(df, numeric_cols):
    """
    Échelle les colonnes numériques en utilisant StandardScaler.
    
    args :
    df : Le DataFrame d'entrée.
    numeric_cols (list) : Liste des colonnes numériques à scaler.
    
    Return :
    pd.DataFrame : DataFrame avec les colonnes numériques mises à l'échelle.
    """
    scaler = StandardScaler()
    df_scaled = df.copy()  # Créer une copie du DataFrame pour éviter de modifier l'original
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    plot_scaling(df_scaled, numeric_cols, 'StandardScaler')  # Placer le plot après le scaling
    return df_scaled

def min_max_scaler(df, numeric_cols):
    """
    Échelle les colonnes numériques en utilisant MinMaxScaler.
    
    PArgs:
    df (pd.DataFrame) : Le DataFrame d'entrée.
    numeric_cols (list) : Liste des colonnes numériques à scaler.
    
    Returns :
    pd.DataFrame : DataFrame avec les colonnes numériques mises à l'échelle.
    """
    scaler = MinMaxScaler()
    df_scaled = df.copy()  # Créer une copie du DataFrame pour éviter de modifier l'original
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    plot_scaling(df_scaled, numeric_cols, 'MinMaxScaler')  # Placer le plot après le scaling
    return df_scaled

def robust_scaler(df, numeric_cols):
    """
    Échelle les colonnes numériques en utilisant RobustScaler.
    
    Args:
    df (pd.DataFrame) : Le DataFrame d'entrée.
    numeric_cols (list) : Liste des colonnes numériques à scaler.
    
    Return :
    pd.DataFrame : DataFrame avec les colonnes numériques mises à l'échelle.
    """
    scaler = RobustScaler()
    df_scaled = df.copy()  # Créer une copie du DataFrame pour éviter de modifier l'original
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    plot_scaling(df_scaled, numeric_cols, 'RobustScaler')  # Placer le plot après le scaling
    return df_scaled

def max_abs_scaler(df, numeric_cols):
    """
    Échelle les colonnes numériques en utilisant MaxAbsScaler.
    
    Args :
    df : Le DataFrame d'entrée.
    numeric_cols : Liste des colonnes numériques à scaler.
    
    Return :
    pd.DataFrame : DataFrame avec les colonnes numériques mises à l'échelle.
    """
    scaler = MaxAbsScaler()
    df_scaled = df.copy()  # Créer une copie du DataFrame pour éviter de modifier l'original
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    plot_scaling(df_scaled, numeric_cols, 'MaxAbsScaler')  # Placer le plot après le scaling
    return df_scaled

def plot_scaling(df, numeric_cols, scaler_name):
    """
     graphique des données avant et après le scaling.
    
    Args :
    df   : Le DataFrame avec les données mises à l'échelle.
    numeric_cols (list) : Liste des colonnes numériques mises à l'échelle.
    scaler_name (str) : Le nom du scaler utilisé.
    """
    plt.figure(figsize=(12, 6))
    for col in numeric_cols:
        plt.subplot(1, len(numeric_cols), numeric_cols.index(col) + 1)
        sns.histplot(df[col], kde=True, color='skyblue')
        plt.title(f'{col} - {scaler_name}')
        plt.xlabel(col)
        plt.ylabel('Fréquence')
    plt.tight_layout()
    plt.show()
