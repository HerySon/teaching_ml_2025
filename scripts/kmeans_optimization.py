from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def optimize_kmeans(df, numeric_cols, param_grid):
    """
    Optimize K-Means hyperparameters using GridSearchCV.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_cols (list): List of numeric column names.
    param_grid (dict): Parameter grid for GridSearchCV.

    Returns:
    dict: Best parameters found by GridSearchCV.
    """
    # Split the data into training and test sets
    X = df[numeric_cols]
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    # Create KMeans model
    kmeans = KMeans(random_state=42)

    # Perform GridSearchCV to find the best parameters
    grid_search = GridSearchCV(kmeans, param_grid, cv=5, scoring='adjusted_mutual_info_score')
    grid_search.fit(X_train)

    # Print the best parameters and score
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")

    return grid_search.best_params_

def find_optimal_clusters(df, numeric_cols, kmeans_kwargs, kmax, kmin=2):
    """
    Find the optimal number of clusters using the elbow or silhouette method.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_cols (list): List of numeric column names.
    kmeans_kwargs (dict): Keyword arguments for KMeans.
    kmax (int): Maximum number of clusters to test.
    kmin (int): Minimum number of clusters to test.

    Returns:
    int: Optimal number of clusters.
    """
    inertia = []
    silhouette_scores = []

    for k in range(kmin, kmax + 1):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df[numeric_cols])
        inertia.append(kmeans.inertia_)
        score = silhouette_score(df[numeric_cols], kmeans.labels_)
        silhouette_scores.append(score)

    # Plot the elbow method
    plt.figure(figsize=(10, 5))
    plt.plot(range(kmin, kmax + 1), inertia, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

    # Plot the silhouette method
    plt.figure(figsize=(10, 5))
    plt.plot(range(kmin, kmax + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

    # Return the optimal number of clusters based on the silhouette score
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + kmin
    return optimal_k

def save_model(model, filename):
    """
    Save the trained model to a file.

    Parameters:
    model: The trained model to save.
    filename (str): The filename to save the model to.
    """
    joblib.dump(model, filename)

def load_model(filename):
    """
    Load a trained model from a file.

    Parameters:
    filename (str): The filename to load the model from.

    Returns:
    model: The loaded model.
    """
    return joblib.load(filename)
