import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def pca_reduction(df, numeric_cols, n_components=2):
    """
    Perform PCA to reduce the number of dimensions.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_cols (list): List of numeric column names.
    n_components (int): Number of principal components to keep.

    Returns:
    pd.DataFrame: DataFrame with reduced dimensions.
    """
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[numeric_cols])

    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(df_scaled)

    pca_cols = []
    for i in range(n_components):
        pca_cols.append(f'PC{i+1}')

    df_pca = pd.DataFrame(df_pca, columns=pca_cols)

    return df_pca

def optimize_pca(df, numeric_cols, target_col, param_grid):
    """
    Optimize the number of PCA components using GridSearchCV.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_cols (list): List of numeric column names.
    target_col (str): Name of the target column.
    param_grid (dict): Parameter grid for GridSearchCV.

    Returns:
    dict: Best parameters found by GridSearchCV.
    """
    X = df[numeric_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best accuracy: {grid_search.best_score_}")

    return grid_search.best_params_

def select_features(df, numeric_cols, n_components=2):
    """
    Select features based on PCA variance explained.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_cols (list): List of numeric column names.
    n_components (int): Number of principal components to keep.

    Returns:
    pd.DataFrame: DataFrame with selected features.
    """
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[numeric_cols])

    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(df_scaled)

    pca_cols = []
    for i in range(n_components):
        pca_cols.append(f'PC{i+1}')

    df_pca = pd.DataFrame(df_pca, columns=pca_cols)

    return df_pca
