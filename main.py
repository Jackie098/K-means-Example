import pandas as pd

iris = pd.read_csv('iris.csv')
init = iris.head()
ult = iris.tail()

print(init)


