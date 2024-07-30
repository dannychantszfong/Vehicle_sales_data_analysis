'''
Author: Tsz Fong Chan
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

#EDA process and data visualisation
# Load the dataset
file_path = r"Data\car_prices_clean.csv"
df = pd.read_csv(file_path)

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Visualize missing values
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# Plot distributions of numeric columns
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns].hist(bins=15, figsize=(15, 10))
plt.suptitle("Distributions of Numeric Columns")
plt.show()

# Pairplot to see relationships between variables
sns.pairplot(df[numeric_columns])
plt.title("Pairplot of Numeric Columns")
plt.show()

# Correlation matrix
numeric_df = df[numeric_columns]
corr_matrix = numeric_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Frequency distribution of categorical columns
categorical_columns = ['make', 'model', 'trim', 'body', 'transmission', 'state']

for col in categorical_columns:
    print(df[col].value_counts())

import matplotlib.pyplot as plt
import seaborn as sns

# Bar plots for categorical columns
for col in categorical_columns:
    plt.figure(figsize=(12, 6))
    sns.countplot(y=df[col], order=df[col].value_counts().index)
    plt.title(f'Frequency distribution of {col}')
    plt.show()

# Cross tabulation of make and model
cross_tab = pd.crosstab(df['make'], df['model'])
print(cross_tab)

# Pivot table of average selling price by make and model
pivot_table = df.pivot_table(values='sellingprice', index='make', columns='model', aggfunc='mean')
print(pivot_table)

# Group by make and calculate average selling price
grouped = df.groupby('make')['sellingprice'].mean().sort_values(ascending=False)
print(grouped)

# Box plot of selling price by make
plt.figure(figsize=(12, 6))
sns.boxplot(x='make', y='sellingprice', data=df)
plt.xticks(rotation=90)
plt.title('Box plot of Selling Price by Make')
plt.show()

