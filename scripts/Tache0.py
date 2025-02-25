import pandas as pd
import numpy as np


def filter_dataframe(df: pd.DataFrame, cat_threshold: int = 10):
    """
    Filters and automatically selects relevant columns from the DataFrame.

    Parameters:
    df : pd.DataFrame - The input DataFrame
    cat_threshold : int - Maximum number of categories to consider a variable as categorical

    Returns:
    dict - A dictionary containing columns categorized by type
    """

    # Separate numerical columns and attempt downcasting
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].apply(pd.to_numeric, downcast='float')

    # Identify categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    ordinal_cols = []
    nominal_cols = []

    for col in cat_cols:
        unique_values = df[col].nunique()
        if unique_values <= cat_threshold:
            ordinal_cols.append(col)
        else:
            nominal_cols.append(col)

    return {
        "numeric": num_cols,
        "ordinal_categorical": ordinal_cols,
        "nominal_categorical": nominal_cols
    }


# Exemple d'utilisation
df = pd.read_csv(
    "/Users/chancybayedi-mayombo/PycharmProjects/Project_ML_ml/teaching_ml_2025/notebooks/openfoodfacts_50000_lignes.csv",
    encoding="utf-8",
    sep="\t",
    low_memory=False
)

filtered_columns = filter_dataframe(df)
print(filtered_columns)
