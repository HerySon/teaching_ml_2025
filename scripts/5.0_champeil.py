'''
Functions related to task 5.0 -> Train and optimize a K-means model
--------------------------------
determine_clusters will test different amount of clusters (from 1 to 9) and show the inertia
to help you determine of much clusters you want to create during the model init
Arguments : Df -> Dataframe to be analysed
Output : Plt -> The plot for the intertia per cluster metric
---------------------------------
train_kmeans will create and train a Kmeans model. Model will be created with the parameters found using GridSearch
Arguments : df -> the dataframe to be used for training
            nb_clusters : number of clusters that the model will create, defaut value is 4 of not given

Output : best_kmeans -> the trained model, to be used as you need
'''
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

def determine_clusters(df):

    # Calculate the sum of the quadratics errors for each numbers of clusters
    inertias = []
    for i in range(1, 10):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(df)
        inertias.append(kmeans.inertia_)

    # Display the curve for inertia/cluster
    plt.plot(range(1, 11), inertias)
    plt.xlabel('Cluster ammount')
    plt.ylabel('Inertia')
    plt.title('Inertia per cluster')
    plt.show()

    return plt


def train_kmeans(df, nb_clusters=4):

    # Initialize the KMeans model
    kmeans = KMeans()

    # Define parameters for GridSearch
    param_grid = {
        'n_clusters': [1,2,3,4,5,6,7,8,9],  # Around the specified nb_clusters
        'init': ['k-means++', 'random'],  # Initialization methods
        'max_iter': [300],  # Classic number of iterations
        'n_init': [10]  # Classic number of restarts
    }

    # GridSearch
    grid_search = GridSearchCV(estimator=kmeans, param_grid=param_grid, cv=3)
    grid_search.fit(df)  # Training with GridSearch

    # Best parameters found
    best_params = grid_search.best_params_

    # Create KMeans model with the best parameters
    grid_kmeans = KMeans(
        n_clusters=best_params['n_clusters'],
        init=best_params['init'],
        max_iter=best_params['max_iter'],
        n_init=best_params['n_init']
    )

    # Train the model with the best parameters
    grid_kmeans.fit(df)

    return grid_kmeans