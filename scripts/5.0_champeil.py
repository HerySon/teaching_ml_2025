'''
Functions related to task 5.0 -> Train and optimize a K-means model
--------------------------------
determine_clusters will test different amount of clusters (from 1 to 9) and show the inertia
to help you determine of much clusters you want to create during the model init
Arguments : Df -> Dataframe to be analysed
Output : Plt -> The plot for the intertia per cluster metric
---------------------------------
train_kmeans will create and train a Kmeans model. Model will be created with the parameters found using GridSearch
Arguments : df -> the dataframe to be used for training, it will be splitted in train and test during the process
            nb_clusters : number of clusters that the model will create, defaut value is 4 of not given

Output : best_kmeans -> the trained model, to be used as you need
'''
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score

def determine_clusters(df):

    # Calculate the sum of the quadratics errors for each numbers of clusters
    inertias = []
    for i in range(1, 10):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(df)
        inertias.append(kmeans.inertia_)

    # Displat the curve for inertia/cluster
    plt.plot(range(1, 11), inertias)
    plt.xlabel('Cluster ammount')
    plt.ylabel('Inertia')
    plt.title('Inertia per cluster')
    plt.show()

    return plt

from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

def train_kmeans(df, nb_clusters=4):

    # Split the data into train and test
    X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

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
    grid_search.fit(X_train)  # Training with GridSearch on training data

    # Best parameters found
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # Create KMeans model with the best parameters
    best_kmeans = KMeans(
        n_clusters=best_params['n_clusters'],
        init=best_params['init'],
        max_iter=best_params['max_iter'],
        n_init=best_params['n_init']
    )

    # Train the model with the best parameters
    best_kmeans.fit(X_train)

    # Make predictions on test data
    #predictions = best_kmeans.predict(X_test)

    return best_kmeans