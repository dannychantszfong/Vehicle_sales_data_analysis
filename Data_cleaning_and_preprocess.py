import pandas as pd
import numpy as np
import os



def initial_statics(df):
    '''
    This function will require data parameter as a DataFrame for future usage, and it will return number of attributes, attribute names and number of data 
    '''
    attr_num = len(df.columns)
    data_num = len(df)
    attr = df.columns

    return attr_num, data_num, attr

def handle_missing_data(df, numeric_method='mean', non_numeric_method='mode', specific_value=None):
    if numeric_method == 'drop' or non_numeric_method == 'drop':
        df = df.dropna()
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

        if numeric_method == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif numeric_method == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif numeric_method == 'value' and specific_value is not None:
            df[numeric_cols] = df[numeric_cols].fillna(specific_value)

        if non_numeric_method == 'mode':
            df[non_numeric_cols] = df[non_numeric_cols].apply(lambda col: col.fillna(col.mode()[0] if not col.mode().empty else col))

    return df

def identify_outliers(df, threshold=3):
    outliers = {}
    for column in df.select_dtypes(include=[np.number]).columns:
        mean = df[column].mean()
        std = df[column].std()
        outliers[column] = df[(df[column] - mean).abs() > threshold * std]
    return outliers

def remove_outliers(df, threshold=3):
    for column in df.select_dtypes(include=[np.number]).columns:
        mean = df[column].mean()
        std = df[column].std()
        df = df[(df[column] - mean).abs() <= threshold * std]
    return df

def correct_data_types(df):
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='ignore')
    return df

def clean_duplicates(df):
    df = df.drop_duplicates()
    return df


# Define the local path
LOCAL_PATH = os.path.dirname(__file__)  # Get local file path

# Load the dataset
file_path = os.path.join(LOCAL_PATH, 'car_prices_ori.csv')
df = pd.read_csv(file_path)

# Initial statistics
attr_num, data_num, attr = initial_statics(df)
print(f"This dataset has {attr_num} attributes and {data_num} rows of data.")
print("The attributes are:", ', '.join(attr))
print()

# Handle missing data
df = handle_missing_data(df, numeric_method='drop', non_numeric_method='drop')

# Identify and remove outliers
'''
outliers = identify_outliers(df)
print("Outliers identified in the dataset:")
for column, outlier_data in outliers.items():
    print(f"Column {column} has {len(outlier_data)} outliers.")
df = remove_outliers(df)
'''
# Correct data types
df = correct_data_types(df)

# Clean duplicates
df = clean_duplicates(df)

# Save the cleaned dataset
cleaned_file_path = os.path.join(LOCAL_PATH, 'car_prices_clean.csv')
df.to_csv(cleaned_file_path, index=False)
print(f'Cleaned dataset saved to: {cleaned_file_path}')