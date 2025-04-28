"""
Functions for the 1.1 Task
the function onehotencoder will create a column for each unique column values among categorical columns
Argument : Df -> any pandas Dataframe
Output : Df -> A pandas Dataframe
"""


from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def onehotencoder(df):

    # Init, the argument 'sparse_output=False' ensure we get a dense matrix
    encoder = OneHotEncoder(sparse_output=False)

    # Apply onehotencoder to all categorical columns
    encoded_columns = encoder.fit_transform(df.select_dtypes(include=['object']))

    # Convert the encoded columns to a DataFrame and assign meaningful column names
    encoded_df = pd.DataFrame(encoded_columns,columns=encoder.get_feature_names_out(df.select_dtypes(include=['object']).columns))

    # Concatenate the encoded columns with the original DataFrame and drop the original categorical columns
    df = pd.concat([df, encoded_df], axis=1).drop(columns=df.select_dtypes(include=['object']).columns)

    return df


