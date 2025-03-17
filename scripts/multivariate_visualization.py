import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_correlation_matrix(df, numeric_cols):
    """
    Plot the correlation matrix as a heatmap.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_cols (list): List of numeric column names.

    Returns:
    None
    """
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix Heatmap')
    plt.show()

def plot_pairwise_scatter(df, numeric_cols):
    """
    Plot pair-wise scatter plots for all pairs of numeric features.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_cols (list): List of numeric column names.

    Returns:
    None
    """
    sns.pairplot(df[numeric_cols])
    plt.suptitle('Pair-wise Scatter Plots', y=1.02)
    plt.show()

def plot_boxplots(df, numeric_cols):
    """
    Plot boxplots for all numeric features.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_cols (list): List of numeric column names.

    Returns:
    None
    """
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(3, 3, i)
        sns.boxplot(y=df[col])
        plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()

def plot_histograms(df, numeric_cols):
    """
    Plot histograms for all numeric features.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_cols (list): List of numeric column names.

    Returns:
    None
    """
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(3, 3, i)
        sns.histplot(df[col], kde=True)
        plt.title(f'Histogram of {col}')
    plt.tight_layout()
    plt.show()
