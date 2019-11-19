import pandas as pd  # Me dá modelos .csv
from sklearn.cluster import KMeans  # permite a execução do algoritmo de cluster

iris = pd.read_csv('iris.csv')  # Puxando arquivo .csv para utilizá-lo
init = iris.head()  # Exibindo os 5 primeiros valores
ult = iris.tail()  # Exibindo os 5 últimos

x = iris.iloc[:, 0:4].values

# Método de inicialização dos clusters
kmeans = KMeans(n_clusters=3, init='random')  # 1º Numero de clusters  2º Forma como ele é gerado

# Executando o algoritmo e agrupando os dados
print("\nFIT -> ")
print (kmeans.fit(x))

# Verifica e mostra os centroids gerados
print("\nCLUSTER_CENTERS_ -> ")
print(kmeans.cluster_centers_)

# Agrupar os dados e retornar uma tabela de distancias. Esta tabela e criada de forma que
# cada instancia contem os valores da distancia em relacao a cada cluster
distance = kmeans.fit_transform(x)
print("\nFIT_TRANSFORM -> ")
print(distance)

# Retorna os labels para cada instancia, ou seja, o codigo do cluster que a instancia de dados
# foi atribuida
labels = kmeans.labels_
print("\nLABELS -> ")
print(labels)
