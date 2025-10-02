import pandas as pd

data = pd.read_csv('DataSet\online_retail_II.csv')

#print(data.isnull().sum())


data = data.drop(columns=['Description'])

#print(data.shape)

data.to_csv('DataSet\online_retail_II_v1.csv', index=False)