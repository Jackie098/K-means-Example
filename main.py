import pandas as pd     #Me dá modelos .csv
from sklearn.cluster import KMeans      #permite a execução do algoritmo de cluster

iris = pd.read_csv('iris.csv')  #Puxando arquivo .csv para utilizá-lo
init = iris.head()              #Exibindo os 5 primeiros valores
ult = iris.tail()           #Exibindo os 5 últimos'

x = iris.iloc[:, 0:4].values

#1º Numero de clusters  2º Forma como ele é gerado
kmeans = KMeans(n_clusters=3, init='random')
print(kmeans.fit(x))       #Executando o algoritmo e agrupando os dados

