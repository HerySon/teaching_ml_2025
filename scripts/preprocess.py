import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.impute import KNNImputer




def clear_missing_data(df):
    """
    Removes empty columns from the DataFrame

    Args:
        df (pandas.DataFrame): DataFrame containing missing values

    Returns:
        pandas.DataFrame: DataFrame without empty columns
    """
    missing_values = df.isnull().sum() / len(df) * 100
  

    empty_columns_count = missing_values[missing_values == 100].count()
    print(f"Nombre de colonnes vides (100% de valeurs manquantes) : {empty_columns_count}")

    df = df.dropna(axis=1, how='all')

    return df



def percent_data(df, threshold):
    """
    Identifies columns with more than 80% missing values

    Args:
        df (pandas.DataFrame): DataFrame containing missing values

    Returns:
        pandas.Series, int: Result with more than 80% missing values
    """
    percent_missing = df.isnull().sum() * 100 / len(df)
    percent_missing.sort_values(ascending=False, inplace=True)

    threshold_view = 2
    filtered = percent_missing[percent_missing.values > threshold_view]

  #  threshold = 80
    columns_to_drop = percent_missing[percent_missing.values > threshold].index
    count_columns_to_drop = len(columns_to_drop)

    columns_to_drop_details = percent_missing[percent_missing.values > threshold]

    return columns_to_drop_details, threshold



def visualize(df, columns_to_drop, threshold):
    """
    Displays a graph of columns with too many missing values

    Args:
        df (pandas.DataFrame): 
        columns_to_drop_details (pandas.Series): 
        threshold (int): 

    Returns:
        pandas.DataFrame: The original DataFrame.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=columns_to_drop.index, y=columns_to_drop.values)
    plt.xticks(rotation=90)
    plt.xlabel('Colonnes')
    plt.ylabel('% de valeurs manquantes')
    plt.title(f"Colonnes avec plus de {threshold}% de valeurs manquantes")
    plt.tight_layout()
    plt.show()

    return df

def delete_data(df,columns_to_drop):
    """_summary_ removes columns specified via columns_to_drop

    Args:
        df (pd.DataFrame): DataFrame to clean
        columns_to_drop (list): List of columns to remove

    Returns:
        _type_: _description_
    """
    df.drop(columns_to_drop, axis='columns', inplace=True)
    return df


def impute_data(df):
    """
    Imputes missing values in numerical columns 

    Args:
        df (pd.DataFrame): input DataFrame

    Returns:
        pd.DataFrame: DataFrame with imputed numerical values
    """
    df_original = df.copy()
    numeric_features = df.select_dtypes(include=['float', 'int'])
    
    imputer = KNNImputer(missing_values=np.nan)
    imputed_values = imputer.fit_transform(numeric_features)
    imputed_values = np.round(imputed_values, 1)

    df.loc[:, numeric_features.columns] = imputed_values
    
    return df_original, df
    

def compare_dist(df_original, df_imputed, feature):
    """
    Compares the distribution of a variable before and after imputing missing values.

    Parameters:
    - df: Original DataFrame
    - df_imputed: DataFrame after imputation
    - feature: Name of the column to analyze

    Returns:
    - The matplotlib figure object for display
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    sns.histplot(df_original.loc[:, feature], kde=True, ax=axes[0])
    axes[0].set_title(f"Raw {feature}")
    axes[0].set_xlim(0, 4000) 


    sns.histplot(df_imputed.loc[:, feature], kde=True, ax=axes[1])
    axes[1].set_title(f"Imputed {feature}")
    axes[1].set_xlim(0, 4000)
 

    return fig