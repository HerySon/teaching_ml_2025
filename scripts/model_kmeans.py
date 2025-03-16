import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
import joblib


def find_clusters_nb(self, method, X, kmeans_kwargs, kmax, kmin=2):
    """
    Determines the optimal number of clusters using the Elbow method or Silhouette score.

    Parameters:
    -----------
    method (str) : The method to use ('elbow' for inertia, 'silhouette' for silhouette score).
    X (array-like) : The dataset used for clustering.
    kmeans_kwargs (dict) : Additional parameters for KMeans.
    kmax (int) : The maximum number of clusters to test.
    kmin (int, optional): The minimum number of clusters to test (default is 2).

    Returns:
    --------
    list : A list of metric values corresponding to each number of clusters tested.
    """
    self.kmax = kmax
    self.kmin = kmin
    self.method = method
    self.metric = []

    if self.method == 'elbow': 
        self.metric_name = "Inertia"
        for k in range(self.kmin, self.kmax+1):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs).fit(X)
            self.metric.append(kmeans.inertia_)

    elif self.method == 'silhouette': 
        self.metric_name = "Silhouette Coefficient"
        for k in range(self.kmin, self.kmax+1):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs).fit(X)
            self.metric.append(silhouette_score(X, kmeans.labels_))

    else:
        raise ValueError("Invalid method. Choose 'elbow' or 'silhouette'.")

    return self.metric
        
def plot_metric(self):
    """
    Plots the computed metric (Inertia or Silhouette score) to visualize the optimal cluster number.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    plt.style.use("seaborn")
    plt.plot(range(self.kmin, self.kmax+1), self.metric, marker='o')
    plt.xticks(range(self.kmin, self.kmax+1))
    plt.xlabel("Number of clusters")
    plt.ylabel(self.metric_name)
    plt.title(f"Optimal cluster search using {self.method} method")
    plt.show()


def optimize_kmeans(X_train, random_state=42):
    """
    Optimizes the hyperparameters of KMeans using GridSearchCV.

    Parameters:
    -----------
    X_train (array-like) : The training dataset for clustering.
    random_state (int), optional : Random seed for KMeans to ensure reproducibility (default is 42).

    Returns:
    --------
    dict : The best hyperparameters found by GridSearchCV.
    """
    param_grid = {
        'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],  # Range of clusters to test
        'init': ['k-means++', 'random'],  # Initialization strategies
        "algorithm":('elkan', 'full'), #K-means algorithm to use
        'max_iter': [100, 300, 500]  # Number of iterations
    }

    kmeans = KMeans(random_state=random_state)
    optimizer = GridSearchCV(kmeans, param_grid, cv=3).fit(X_train)
    
    return optimizer.best_params_



def train_kmeans(best_params, X_train, n_clusters=None, random_state=42, model_path="kmeans_model.pkl"):
    """
    Trains a KMeans model using the best parameters from GridSearchCV, 
    with an optional override for the number of clusters.

    Parameters:
    -----------
    best_params (dict) : Dictionary of best hyperparameters obtained from GridSearchCV. 
    X_train (array-like) : The dataset used to train the KMeans model.
    n_clusters (int, optional) : If provided, overrides the 'n_clusters' value in best_params.
    random_state (int, optional) : Random seed for reproducibility (default is 42).
    model_path (str, optional) : Path to save the trained KMeans model (default is "kmeans_model.pkl").

    Returns:
    --------
    KMeans : The trained KMeans model.
    """

    # Use best_params but allow overriding n_clusters
    kmeans_params = best_params.copy()
    if n_clusters is not None:
        kmeans_params['n_clusters'] = n_clusters

    # Train KMeans with the selected parameters
    kmeans = KMeans(**kmeans_params, random_state=random_state)
    kmeans.fit(X_train)

    # Save the model
    joblib.dump(kmeans, "kmeans_model.pkl")

    return kmeans
