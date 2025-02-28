import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib

def select_numeric_features(df):
    """
    Sélectionne uniquement les colonnes numériques du DataFrame.
    
    Paramètres :
    ------------
    df (pandas.DataFrame) : Données brutes.
    
    Retourne :
    ---------
    pandas.DataFrame : Sous-ensemble des données contenant uniquement les colonnes numériques.
    """
    return df.select_dtypes(include=['float64', 'int64'])

def create_kmeans_pipeline(n_clusters, n_components=None):
    """
    Crée un pipeline intégrant la normalisation, la PCA et le clustering KMeans.
    
    Paramètres :
    ------------
    n_clusters (int) : Nombre de clusters à utiliser pour KMeans.
    n_components (int, optionnel) : Nombre de composants pour la PCA. Si None, automatique.
    
    Retourne :
    ---------
    sklearn.pipeline.Pipeline : Pipeline de transformation et clustering.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components)),
        ('kmeans', KMeans(n_clusters=n_clusters, random_state=42, n_init='auto'))
    ])

def calculate_pca_components(data, variance_threshold=0.95):
    """
    Calcule le nombre optimal de composants PCA en fonction de la variance expliquée.
    
    Paramètres :
    ------------
    data (array-like) : Données à analyser.
    variance_threshold (float) : Seuil de variance expliquée (par défaut 0.95 pour 95%).
    
    Retourne :
    ---------
    int : Nombre optimal de composants PCA.
    """
    pca_temp = PCA().fit(data)
    cumulative_variance = np.cumsum(pca_temp.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    return n_components

def determine_optimal_clusters(method, data, kmeans_params, max_k, min_k=2):
    """
    Identifie le nombre idéal de clusters via la méthode Elbow ou le score Silhouette.
    
    Paramètres :
    ------------
    method (str) : Méthode utilisée ('elbow' ou 'silhouette').
    data (array-like) : Données à analyser.
    kmeans_params (dict) : Paramètres pour KMeans.
    max_k (int) : Nombre maximum de clusters à tester.
    min_k (int, optionnel) : Nombre minimum de clusters à tester (par défaut 2).
    
    Retourne :
    ---------
    list : Liste des scores calculés pour chaque nombre de clusters.
    """
    scores = []
    metric_label = "Inertie" if method == 'elbow' else "Score de silhouette"
    
    for k in range(min_k, max_k + 1):
        pipeline = create_kmeans_pipeline(k)
        pipeline.fit(data)
        labels = pipeline.named_steps['kmeans'].labels_
        scores.append(pipeline.named_steps['kmeans'].inertia_ if method == 'elbow' else silhouette_score(data, labels))
    
    afficher_evolution_clusters(min_k, max_k, scores, metric_label, method)
    return scores, metric_label

def afficher_evolution_clusters(min_k, max_k, scores, metric_label, method):
    """
    Génère un graphique montrant l'évolution de la métrique en fonction du nombre de clusters.
    """
    plt.style.use("seaborn")
    plt.plot(range(min_k, max_k + 1), scores, marker='o')
    plt.xticks(range(min_k, max_k + 1))
    plt.xlabel("Nombre de clusters")
    plt.ylabel(metric_label)
    plt.title(f"Analyse du nombre optimal de clusters avec la méthode {method}")
    plt.show()

def rechercher_meilleurs_parametres(X_train, random_state=42):
    """
    Recherche les meilleurs hyperparamètres pour KMeans via GridSearchCV.
    """
    param_grid = {
        'n_clusters': list(range(2, 11)),
        'init': ['k-means++', 'random'],
        'algorithm': ['elkan', 'full'],
        'max_iter': [100, 300, 500]
    }
    
    model = KMeans(random_state=random_state, n_init='auto')
    recherche = GridSearchCV(model, param_grid, cv=3).fit(X_train)
    
    return recherche.best_params_

def entrainer_kmeans(best_params, X_train, n_clusters=None, random_state=42, model_path=None):
    """
    Entraîne un modèle KMeans avec les meilleurs paramètres trouvés.
    """
    params = best_params.copy()
    if n_clusters is not None:
        params['n_clusters'] = n_clusters
    
    model = KMeans(**params, random_state=random_state, n_init='auto')
    model.fit(X_train)
    
    # Sauvegarde dynamique du modèle
    if model_path is None:
        model_path = f"kmeans_model_{params['n_clusters']}_{params['init']}.pkl"
    
    joblib.dump(model, model_path)
    return model
