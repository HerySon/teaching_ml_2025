from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

def optimize_dbscan_params(X):
    """
    Optimize DBSCAN hyperparameters using GridSearchCV.

    Parameters:
        X (numpy array or DataFrame): The dataset on which to optimize DBSCAN.

    Returns:
        dict: Best parameters for DBSCAN.
    """
    param_grid = {
        'eps': [0.2, 0.3, 0.4, 0.5],  # Distance threshold
        'min_samples': [5, 10, 15]   # Minimum points in a neighborhood
    }
    
    dbscan = DBSCAN()
    grid_search = GridSearchCV(dbscan, param_grid, scoring='silhouette', cv=3)
    grid_search.fit(X)
    
    return grid_search.best_params_

def train_dbscan(X, best_params):
    """
    Train a DBSCAN model with given hyperparameters.

    Parameters:
        X (numpy array or DataFrame): The dataset to cluster.
        best_params (dict): Best hyperparameters from optimization.

    Returns:
        DBSCAN: Trained DBSCAN model.
        numpy array: Cluster labels.
    """
    db = DBSCAN(**best_params).fit(X)
    labels = db.labels_
    return db, labels

def plot_dbscan_clusters(X, labels):
    """
    Plot DBSCAN clustering results.

    Parameters:
        X (numpy array or DataFrame): The dataset used for clustering.
        labels (numpy array): Cluster labels assigned by DBSCAN.
    """
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # Black for noise

        class_member_mask = labels == k
        xy = X[class_member_mask]
        
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], edgecolors="k", s=50 if k != -1 else 20)

    plt.title(f"DBSCAN Clustering - Estimated Clusters: {len(unique_labels) - (1 if -1 in labels else 0)}")
    plt.show()
