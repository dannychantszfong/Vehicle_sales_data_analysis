import pandas as pd
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("Data\car_prices_clean.csv")
print(df)
print(df.describe())


scaler = StandardScaler()
df[["year_T", "condition_T","odometer_T","mmr_T","sellingprice_T"]] = scaler.fit_transform(df[["year","condition","odometer","mmr","sellingprice"]])

print(df)

def optimise_k_means(data, max_k):
    means = []
    intertias = []

    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k,n_init=10)
        kmeans.fit(data)

        means.append(k)
        intertias.append(kmeans.inertia_)
    
    #fig = plt.subplot(figsize=(10,5))
    plt.plot(means,intertias, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Intertia')
    plt.grid(True)
    plt.savefig("Img\Cluster_Analysis_Img\optimise_k_means.png")
    plt.show()

optimise_k_means(df[["year_T", "condition_T","odometer_T","mmr_T","sellingprice_T"]],10)


kmeans = KMeans(n_clusters = 3,n_init=10)
kmeans.fit(df[["year_T", "condition_T","odometer_T","mmr_T","sellingprice_T"]])
df['kmeans_3'] = kmeans.labels_

plt.scatter(x = df['year'], y = df['condition'], c = df['kmeans_3'])
plt.savefig("Img\Cluster_Analysis_Img\scatter.png")
plt.show()




