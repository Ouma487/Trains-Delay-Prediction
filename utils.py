import pandas as pd
import re
import sklearn
import numpy as np

#Functions to split the data according to the date
def select_test_data(data):
    data['date'] = data['date'].str.strip()  # Remove leading/trailing spaces if any

    # Define the regular expression pattern
    pattern = re.compile(r'^2023-\d{2}$', re.IGNORECASE)

    # Filter rows with the desired date format
    test_data = data[data['date'].str.contains(pattern, na=False)]
    return test_data

def select_train_data(data,test_data):
    # Find the indices of rows in 'data' that are not in 'test_data'
    indices_to_keep = ~data.index.isin(test_data.index)

    # Create the 'train_data' DataFrame by selecting rows from 'data' using the indices to keep
    train_data = data[indices_to_keep].reset_index(drop=True)
    return train_data

#Function to remove the extra columns 
def clean_data(data,columns):
    data=data.drop(columns,axis=1)
    return data

# Function to get the target values
def target_fun(data,columns):
    return data[columns].values.ravel() #.ravel() pour aplatir en un vecteur

# Function to get the features
def features(data, target_columns):
    # Use the drop method to remove the target columns from the dataset
    features_data = data.drop(columns=target_columns)
    
    return features_data
