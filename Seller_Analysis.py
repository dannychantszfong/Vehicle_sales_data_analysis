import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("Data\car_prices_clean.csv")

# Convert 'saledate' to datetime
df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce')

# Calculate the average selling price for each seller
seller_avg_price = df.groupby('seller')['sellingprice'].mean().sort_values()

# Calculate the total sales volume for each seller
seller_sales_volume = df['seller'].value_counts()

# Plot the average selling prices by seller
plt.figure(figsize=(12, 8))
sns.barplot(x=seller_avg_price.index, y=seller_avg_price.values)
plt.xticks(rotation=90)
plt.xlabel('Seller')
plt.ylabel('Average Selling Price')
plt.title('Average Selling Price by Seller')
plt.savefig("Img\Seller_Analysis_Img\Average Selling Price by Seller.png")
plt.show()

# Plot the sales volumes by seller
plt.figure(figsize=(12, 8))
sns.barplot(x=seller_sales_volume.index, y=seller_sales_volume.values)
plt.xticks(rotation=90)
plt.xlabel('Seller')
plt.ylabel('Sales Volume')
plt.title('Sales Volume by Seller')
plt.savefig("Img\Seller_Analysis_Img\Sales Volume by Seller.png")
plt.show()

# Combine average selling price and sales volume into a single DataFrame
seller_performance = pd.DataFrame({
    'Average Selling Price': seller_avg_price,
    'Sales Volume': seller_sales_volume
}).sort_values(by='Sales Volume', ascending=False)

# Plot combined performance of sellers
fig, ax1 = plt.subplots(figsize=(12, 8))

color = 'tab:blue'
ax1.set_xlabel('Seller')
ax1.set_ylabel('Average Selling Price', color=color)
ax1.bar(seller_performance.index, seller_performance['Average Selling Price'], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.tick_params(axis='x', rotation=90)

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Sales Volume', color=color)
ax2.plot(seller_performance.index, seller_performance['Sales Volume'], color=color, marker='o')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Seller Performance: Average Selling Price and Sales Volume')
plt.savefig("Img\Seller_Analysis_Img\Seller Performance_Average Selling Price and Sales Volume.png")
plt.show()

# Print seller performance data
print(seller_performance)
