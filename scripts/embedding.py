import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

def apply_pca(X, n_components=None):
    """
    Apply PCA on the dataset.

    Parameters:
        X (numpy array or DataFrame): The dataset to transform.
        n_components (int or None): Number of components to keep. If None, keep all.

    Returns:
        PCA: Trained PCA model.
        numpy array: Transformed dataset.
    """
    n_components = n_components if n_components else X.shape[1]  # Default: all components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return pca, X_pca

def optimize_pca_components(X, variance_threshold=0.95):
    """
    Determine the optimal number of PCA components to retain.

    Parameters:
        X (numpy array or DataFrame): The dataset.
        variance_threshold (float): Minimum cumulative variance to retain.

    Returns:
        int: Optimal number of components.
    """
    pca = PCA().fit(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    optimal_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    return optimal_components

def plot_pca_variance(X, pca):
    """
    Plot the explained variance ratio of each principal component.

    Parameters:
        X (numpy array or DataFrame): The dataset to analyze.
        pca (PCA): The trained PCA model.  
    """

    plt.plot(range(1, X.shape[1] + 1), pca.explained_variance_ratio_, marker="o", markerfacecolor="r")
    plt.xlabel('Principal Components')
    plt.xticks(range(1, X.shape[1] + 1))
    plt.ylabel('Explained Variance Ratio')
    plt.grid(linestyle='--')
    plt.title("Scree Plot - Explained Variance")
    plt.show()

def select_important_features(X, y, k=10):
    """
    Select the top k most important features using ANOVA F-test.

    Parameters:
        X (numpy array or DataFrame): Feature dataset.
        y (numpy array or Series): Target variable.
        k (int): Number of features to keep.

    Returns:
        numpy array: Reduced dataset with top k features.
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    return X_selected


def plot_pca_variable_contribution(pca, X):
    """
    Plot the contribution of each variable to the first two principal components.

    Parameters:
        pca (PCA): The trained PCA model.
        X (numpy array or DataFrame): Feature dataset.

    Returns:
        None
    """
    pcs = pca.components_
    fig, ax = plt.subplots(figsize=(14, 14))

    for i, (x, y) in enumerate(zip(pcs[0, :], pcs[1, :])):
        # Draw a segment from origin to (x, y)
        ax.plot([0, x], [0, y], color='k')
        # Display feature name
        plt.text(x, y, X.columns[i], fontsize=14)

    # Plot horizontal and vertical reference lines
    plt.plot([-0.7, 0.7], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-0.7, 0.7], color='grey', ls='--')

    ax.set_xlabel("Principal Component 1", fontsize=16)
    ax.set_ylabel("Principal Component 2", fontsize=16)
    ax.set_title("Contribution of Variables to Principal Components", fontsize=16)
    plt.show()
