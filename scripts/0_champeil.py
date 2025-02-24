"""
Fonction for the task 0, will show numerical and categorical columns
Will also reduce the size of floats to increase performances
Args : df -> the dataframe to be modified
Return : df -> the modified dataframe
"""
import pandas as pd
import numpy as np
def get_types(df) :

    return df.dtypes

def get_numerical(df):
    # store numerical columns in an array
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    return numeric_columns

def get_categorical(df):
    # store categorical columns in an array -> we saw with df.dtypes that non numerical columns are "object"
    categorical_columns = df.select_dtypes(include=[object]).columns
    return categorical_columns

def downcast_floats(df):
    # Reduce float size to increase performance and reduce dataframe size
    # Target type can be ajusted from float32 to any that suits
    for col in df.select_dtypes(include=[np.float64]).columns:
        df[col] = df[col].astype(np.float32)
    return df

'''
#Playground for testing functions

path="../data/dataset.csv"
df = pd.read_csv(path)
print(get_types(df))
print(get_numerical(df))
print(get_categorical(df))

df=downcast_floats(df)
print(get_types(df))

'''