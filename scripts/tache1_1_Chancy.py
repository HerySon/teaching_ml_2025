# Import des librairies
import pandas as pd
import numpy as np
import missingno as mnso
import matplotlib.pyplot as plt
import seaborn as sns

# Import des modules de scikit-learn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import MissingIndicator, KNNImputer, SimpleImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def encoding_ordered_categorical(df_params, categorical_feature):
    """
    Encodes ordered categorical variables in a DataFrame into numerical values.

    This function uses the OrdinalEncoder from sklearn to transform the specified columns
    into ordinal values, assuming a natural order between categories.

    Args:
        df_params (pd.DataFrame): The DataFrame containing the categorical variables to encode.
        categorical_feature (list of str): List of column names corresponding to ordered categorical variables.

    Returns:
        pd.DataFrame: The updated DataFrame with encoded columns.
    """
    # Vérifier que les colonnes existent dans le DataFrame
    categorical_feature = [col for col in categorical_feature if col in df_params.columns]
    if not categorical_feature:
        print("Aucune colonne catégorielle ordonnée trouvée pour l'encodage.")
        return df_params

    # Créer l'encodeur et transformer les colonnes spécifiées
    oe = OrdinalEncoder()
    df_params[categorical_feature] = oe.fit_transform(df_params[categorical_feature])

    return df_params


def encoding_unordered_categorical(df_params, categorical_feature):
    """
    Encodes unordered categorical variables in a DataFrame using OneHotEncoder.

    This function applies one-hot encoding to the specified categorical columns, converting
    each unique category into a separate binary feature. Unknown categories encountered
    during transformation are ignored.

    Args:
        df_params (pd.DataFrame): The DataFrame containing categorical variables to encode.
        categorical_feature (list of str): List of column names corresponding to unordered categorical variables.

    Returns:
        pd.DataFrame: A new DataFrame with categorical values replaced by one-hot encoded features.
    """
    # Vérifier que les colonnes existent dans le DataFrame
    categorical_feature = [col for col in categorical_feature if col in df_params.columns]
    if not categorical_feature:
        print("Aucune colonne catégorielle non ordonnée trouvée pour l'encodage.")
        return df_params

    # Initialiser OneHotEncoder
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(df_params[categorical_feature])
    encoded_features = enc.transform(df_params[categorical_feature])
    encoded_feature_names = enc.get_feature_names_out(categorical_feature)
    df_encoded_categorical = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df_params.index)

    # Retourner un DataFrame avec les nouvelles colonnes encodées
    df_final = pd.concat([df_params.drop(columns=categorical_feature), df_encoded_categorical], axis=1)
    return df_final


if __name__ == "__main__":
    # Création d'un DataFrame d'exemple
    data = {
        'Category_Ordered': ['Low', 'Medium', 'High', 'Low', 'High'],
        'Category_Unordered': ['Red', 'Blue', 'Green', 'Blue', 'Red'],
        'Value': [1, 2, 3, 4, 5]
    }
    df = pd.DataFrame(data)
    print("DataFrame original:\n", df)

    # Encodage des catégories ordonnées
    df = encoding_ordered_categorical(df, ['Category_Ordered'])
    print("\nAprès encodage des catégories ordonnées:\n", df)

    # Encodage des catégories non ordonnées
    df = encoding_unordered_categorical(df, ['Category_Unordered'])
    print("\nAprès encodage des catégories non ordonnées:\n", df)