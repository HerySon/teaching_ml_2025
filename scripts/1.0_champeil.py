"""
Function for task 1.0
The function will first delete duplicate and empty rows
Then it will delete >80% empty columns column
Finally, it will replace the NaN with nearest neighbour for numericals and most frequent for categoricals
Args : df -> the dataframe to be cleansed
Returns : df -> the cleansed dataframe
"""
from sklearn.impute import SimpleImputer
def clean_dataset(df):

    # Delete the duplicates rows
    df.drop_duplicates(inplace=True)

    # ------------------------------------
    #Delete the empty rows

    if  df.isnull().all().sum() !=0:
        df.dropna(axis=1, how='all', inplace=True)
    #If thre is no empty rows, no operation performed
    else:
        print("No empty rows")

    # ------------------------------------

    #Delete columns with >80% empty rows

    percent_missing = df.isnull().sum() * 100 / len(df) # Calculate how empty each columns are
    percent_missing.sort_values(ascending=False, inplace=True) # Order them

    threshold = 80 # Adjust this number to change the rows emptiness %

    filtered = percent_missing[percent_missing.values > threshold] # Keep only columns where empty% > Threshold
    columns_to_drop = percent_missing[
        percent_missing.values > threshold].index # Keep only the column name, we drop the percentages for now

    df.drop(columns_to_drop, axis='columns', inplace=True) # Drop the columns

    # ------------------------------------

    #Imputation of numerical variables

    numeric_features = df.select_dtypes(include=['float', 'int']) # Keep only the numericals
    imputation = KNNImputer(missing_values=np.nan) # Apply the imputation to NaN values
    imputed = imputation.fit_transform(numeric_features) #Replace the NaN with the nearest neighbour value
    df.loc[:, numeric_features.columns] = imputed # Save the calculted values

    # ------------------------------------

    # Imputation of categorical columns with most frequant value

    # Select the categorical values
    non_numeric_features = df.select_dtypes(exclude=['float', 'int'])

    # Impute with the most frequant value
    imputer_non_numeric = SimpleImputer(strategy='most_frequent') #Most frequent
    imputed_non_numeric = imputer_non_numeric.fit_transform(non_numeric_features) #Calculate the values from NaN to most frequent
    df.loc[:, non_numeric_features.columns] = imputed_non_numeric # Replace the values

    return df # Return the cleaned dataframe