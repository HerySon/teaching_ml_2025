import pandas as pd

def detect_and_filter_columns(df, max_categories=10):
    """
    Processes a DataFrame to detect, filter, and select relevant columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        max_categories (int, optional): Maximum number of unique categories for a categorical column to be retained. Defaults to 10.

    Returns:
        pd.DataFrame: Filtered DataFrame with relevant columns.

    Steps:
    1. Detects and downcasts numerical columns.
    2. Identifies and converts categorical columns.
    3. Filters categorical columns based on unique category count.
    4. Selects and returns relevant columns
    """
    # Detect numerical columns
    numeric_columns = df.select_dtypes(include=['int', 'float']).columns.tolist()

    # Downcast numerical columns if possible
    for col in numeric_columns:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        else:
            df[col] = pd.to_numeric(df[col], downcast='integer')

    # Detect ordinal categorical columns
    ordinal_columns = df.select_dtypes(include='category').columns.tolist()

    # Detect non-ordinal categorical columns
    non_ordinal_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Convert non-ordinal categorical columns to 'category' type
    df[non_ordinal_columns] = df[non_ordinal_columns].astype('category')

    # Filter categorical columns based on the number of unique categories
    categorical_columns = ordinal_columns + non_ordinal_columns
    filtered_categorical_columns = [col for col in categorical_columns if df[col].nunique() <= max_categories]

    # Select relevant columns
    relevant_columns = numeric_columns + filtered_categorical_columns
    df_filtered = df[relevant_columns]

    return df_filtered
 

