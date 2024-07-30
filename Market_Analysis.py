import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("Data\car_prices_clean.csv")

# Convert 'saledate' to datetime
df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce')

# Market Analysis: Compare selling prices across different makes and models
# Analyze the impact of car condition on selling price

# Group by 'make' and calculate the mean selling price for each make
make_avg_price = df.groupby('make')['sellingprice'].mean().sort_values()

# Plot the average selling prices across different makes
plt.figure(figsize=(12, 8))
sns.barplot(x=make_avg_price.index, y=make_avg_price.values)
plt.xticks(rotation=90)
plt.xlabel('Make')
plt.ylabel('Average Selling Price')
plt.title('Average Selling Price by Make')
plt.savefig("Img\Market_Analysis_Img\Average_Selling_Price_by_Make.png")
plt.show()

# Group by 'model' and calculate the mean selling price for each model
model_avg_price = df.groupby('model')['sellingprice'].mean().sort_values()

# Plot the average selling prices across different models
plt.figure(figsize=(12, 8))
sns.barplot(x=model_avg_price.index, y=model_avg_price.values)
plt.xticks(rotation=90)
plt.xlabel('Model')
plt.ylabel('Average Selling Price')
plt.title('Average Selling Price by Model')
plt.savefig("Img\Market_Analysis_Img\Average Selling Price by Model.png")
plt.show()

# Analyze the impact of car condition on selling price
plt.figure(figsize=(12, 8))
sns.boxplot(x='condition', y='sellingprice', data=df)
plt.xlabel('Condition')
plt.ylabel('Selling Price')
plt.title('Impact of Car Condition on Selling Price')
plt.savefig("Img\Market_Analysis_Img\Impact_of_Car_Condition_on_Selling_Price.png")
plt.show()

# Additional analysis: Scatter plot of selling price vs. condition with a trend line
plt.figure(figsize=(12, 8))
sns.scatterplot(x='condition', y='sellingprice', data=df)
sns.lineplot(x='condition', y='sellingprice', data=df, ci=None, color='red')
plt.xlabel('Condition')
plt.ylabel('Selling Price')
plt.title('Selling Price vs. Condition')
plt.savefig("Img\Market_Analysis_Img\Selling_Price_vs._Condition.png")
plt.show()

# Descriptive statistics for selling price by condition
condition_price_stats = df.groupby('condition')['sellingprice'].describe()
print(condition_price_stats)
