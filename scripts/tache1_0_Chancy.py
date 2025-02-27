# Import des librairies
import pandas as pd
import numpy as np
import missingno as mnso
import matplotlib.pyplot as plt
import seaborn as sns  # Correction de la majuscule sur "Import"

# Import des modules de scikit-learn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import MissingIndicator, KNNImputer, SimpleImputer, IterativeImputer  # Correction de "sklern" → "sklearn"


def removal_of_duplicates(df_params):
    """
    This function takes a DataFrame as a parameter and removes duplicate rows
    based on all columns. It returns the DataFrame without duplicates.

    Parameters:
    df_params (DataFrame): The input DataFrame from which duplicates will be removed.

    Returns:
    DataFrame: The DataFrame after removing duplicate rows.
    """
    # Remove duplicates across all columns
    df_params.drop_duplicates(inplace=True)
    return df_params




def data_cleaning(df_params):
    """
    This function takes a DataFrame as a parameter, removes variables (columns)
    with a percentage of missing values greater than 79%, and returns a cleaned DataFrame.

    Parameters:
    df_params (DataFrame): The input DataFrame to be cleaned.

    Returns:
    DataFrame: The DataFrame after removing columns with excessive missing values.
    """
    # Count the number of NaN values per column and convert this number into a percentage of missing values
    percent_missing = df_params.isnull().sum() * 100 / len(df_params)
    # Sort columns by the percentage of missing values in descending order
    percent_missing.sort_values(ascending=False, inplace=True)
    # Threshold for removal in %
    threshold = 79
    # List of columns to drop
    columns_to_drop = percent_missing[percent_missing.values > threshold].index
    # Remove columns with more than 79% missing values
    df_params.drop(columns_to_drop, axis='columns', inplace=True)
    return df_params

def non_useful_columns(df_params):
    """
    This function takes a DataFrame as a parameter, removes specific non-useful columns,
    and returns a cleaned DataFrame.

    Parameters:
    df_params (DataFrame): The input DataFrame from which unnecessary columns will be removed.

    Returns:
    DataFrame: The DataFrame after removing the specified columns.
    """
    colonnes_a_supprimer = [
        "url", "created_t", "created_datetime", "last_modified_t", "last_modified_datetime",
        "last_modified_by", "last_updated_t", "brands_tags", "last_updated_datetime",
        "countries_tags", "countries_en", "states_tags", "states_en", "image_url",
        "image_small_url", "image_nutrition_url", "image_nutrition_small_url"
    ]

    df_params = df_params.drop(columns=colonnes_a_supprimer)
    return df_params


def imputation_of_categorical_val(df_params):
    """
    This function imputes missing values in categorical variables using the most frequent value.

    Parameters:
    df_params (DataFrame): The input DataFrame containing categorical variables with missing values.

    Returns:
    DataFrame: The DataFrame after imputing missing categorical values.
    """
    # Select categorical columns (type object or category)
    categorical_cols = df_params.select_dtypes(include=["object", "category"]).columns
    # Impute missing values with the most frequent value in each column
    imputer = SimpleImputer(strategy="most_frequent")
    df_params[categorical_cols] = imputer.fit_transform(df_params[categorical_cols])
    return df_params


def imputation_of_numerical_val(df_params):
    """
    This function imputes missing values in numerical variables using the k-nearest neighbors (KNN) imputation method.

    Parameters:
    df_params (DataFrame): The input DataFrame containing numerical variables with missing values.

    Returns:
    DataFrame: The DataFrame after imputing missing numerical values.
    """
    # Select numerical columns (type float or int)
    numeric_features = df_params.select_dtypes(include=['float', 'int'])
    # Impute missing values using K-Nearest Neighbors
    imputer = KNNImputer(missing_values=np.nan)
    imputed_values = imputer.fit_transform(numeric_features)
    # Replace the original numerical columns with the imputed values
    df_params.loc[:, numeric_features.columns] = imputed_values
    return df_params




