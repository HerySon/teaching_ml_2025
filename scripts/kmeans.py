import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
import joblib

def select_features(df):
    """
    Select relevant features from the DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing the data

    Returns:
        pandas.DataFrame: DataFrame with selected numeric features
    """
    return df.select_dtypes(include=['float64', 'int64'])

def create_pipeline(n_clusters):
    """
    Create a pipeline for KMeans clustering.

    Args:
        n_clusters (int): Number of clusters for KMeans

    Returns:
        sklearn.pipeline.Pipeline: Pipeline with scaling, PCA, and KMeans steps
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=10)),
        ('kmeans', KMeans(n_clusters=n_clusters, random_state=42))
    ])

def optimize_kmeans(df):
    """
    Optimize KMeans hyperparameters using the Elbow Method and Silhouette Score.

    Args:
        df (pandas.DataFrame): DataFrame containing the data

    Returns:
        tuple: Best pipeline, best number of clusters, range of clusters, SSE, silhouette scores
    """
    range_n_clusters = range(2, 11)
    best_score = -1
    best_n_clusters = 2
    best_pipeline = None

    sse = []
    silhouette_scores = []

    for n_clusters in range_n_clusters:
        pipeline = create_pipeline(n_clusters)
        pipeline.fit(df)
        labels = pipeline.named_steps['kmeans'].labels_
        score = silhouette_score(df, labels)

        sse.append(pipeline.named_steps['kmeans'].inertia_)
        silhouette_scores.append(score)

        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
            best_pipeline = pipeline

    return best_pipeline, best_n_clusters, range_n_clusters, sse, silhouette_scores

def save_model(model, file_path):
    """
    Save the trained model to a file.

    Args:
        model: Trained model to save
        file_path (str): Path where the model will be saved
    """
    joblib.dump(model, file_path)
