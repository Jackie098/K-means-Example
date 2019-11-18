import pandas as pd

iris = pd.read_csv('iris.csv')  #Puxando arquivo .csv para utilizá-lo
init = iris.head()              #Exibindo os 5 primeiros valores
ult = iris.tail()           #Exibindo os 5 últimos

x = iris.iloc[:,0:4].values

print(x)


