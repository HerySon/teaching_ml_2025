import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_correlation_matrix(df, numeric_cols):
    """
    Affiche la matrice de corrélation sous forme de heatmap.

    Paramètres :
    df (pd.DataFrame) : Le DataFrame d'entrée.
    numeric_cols (list) : Liste des noms de colonnes numériques.

    Retourne :
    None
    """
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matrice de Corrélation Heatmap')
    plt.show()

def plot_pairwise_scatter(df, numeric_cols):
    """
    Affiche les nuages de points pour toutes les paires de variables numériques.

    Paramètres :
    df (pd.DataFrame) : Le DataFrame d'entrée.
    numeric_cols (list) : Liste des noms de colonnes numériques.

    Retourne :
    None
    """
    sns.pairplot(df[numeric_cols])
    plt.suptitle('Nuages de Points Pair-wise', y=1.02)
    plt.show()

def plot_boxplots(df, numeric_cols, cols_per_row=3):
    """
    Affiche les boxplots pour toutes les variables numériques.

    Paramètres :
    df (pd.DataFrame) : Le DataFrame d'entrée.
    numeric_cols (list) : Liste des noms de colonnes numériques.
    cols_per_row (int) : Nombre de boxplots par ligne. (par défaut 3)

    Retourne :
    None
    """
    n_cols = len(numeric_cols)
    n_rows = np.ceil(n_cols / cols_per_row).astype(int)
    
    plt.figure(figsize=(cols_per_row * 5, n_rows * 5))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(n_rows, cols_per_row, i)
        sns.boxplot(y=df[col])
        plt.title(f'Boxplot de {col}')
    
    plt.tight_layout()
    plt.show()

def plot_histograms(df, numeric_cols, cols_per_row=3):
    """
    Affiche les histogrammes pour toutes les variables numériques.

    Paramètres :
    df (pd.DataFrame) : Le DataFrame d'entrée.
    numeric_cols (list) : Liste des noms de colonnes numériques.
    cols_per_row (int) : Nombre d'histogrammes par ligne. (par défaut 3)

    Retourne :
    None
    """
    n_cols = len(numeric_cols)
    n_rows = np.ceil(n_cols / cols_per_row).astype(int)

    plt.figure(figsize=(cols_per_row * 5, n_rows * 5))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(n_rows, cols_per_row, i)
        sns.histplot(df[col], kde=True)
        plt.title(f'Histogramme de {col}')
    
    plt.tight_layout()
    plt.show()

# Exemple d'utilisation
# En supposant que 'df' soit votre DataFrame et 'numeric_cols' soit la liste des colonnes numériques que vous voulez visualiser
numeric_cols = ['column1', 'column2', 'column3']  # remplacez par vos noms de colonnes réels
plot_correlation_matrix(df, numeric_cols)
plot_pairwise_scatter(df, numeric_cols)
plot_boxplots(df, numeric_cols)
plot_histograms(df, numeric_cols)
