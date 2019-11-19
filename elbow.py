import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


iris = pd.read_csv('iris.csv')

x = iris.iloc[:, 0:4].values

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='random')
    kmeans.fit(x)
    print(i, kmeans.inertia_)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('O método Elbow')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')          #within cluster sum of squares
plt.show()