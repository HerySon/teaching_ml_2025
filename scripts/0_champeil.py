"""
Fonction for the task 0, will show numerical and categorical columns
Will also reduce the size of floats to increase performances
Args : df -> the dataframe to be modified
Return : df -> the modified dataframe
"""

def filter_df(df) :

    # Check the columns type

    df.dtypes

    # Display numerical columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    print(numeric_cols)

    # Display categorical columns -> we saw with df.dtypes that non numerical columns are "object"

    categorical_columns = df.select_dtypes(include=[object]).columns
    print(categorical_columns)

    # Reduce float size to increase performance and reduce dataframe size
    # Target type can be ajusted from float32 to any that suits
    for col in df.select_dtypes(include=[np.float64]).columns:
        df[col] = df[col].astype(np.float32)

    # No equivalent for int type in this project because there is no int type column besides the row number,code, and timestamps, which seems irrelevent for training

    return df
