import pandas as pd

# Load the dataset
# Replace 'your_dataset.csv' with the path to your CSV file
df = pd.read_csv('Data\car_prices_clean.csv')

# Display the first few rows of the dataframe to understand its structure
print(df.head())

# Get descriptive statistics
desc_stats = df.describe()

# Print descriptive statistics
print(desc_stats)

desc_stats = df.describe(percentiles=[.1, .25, .5, .75, .9])
print(desc_stats)

cat_desc_stats = df.describe(include=['object'])
print(cat_desc_stats)
