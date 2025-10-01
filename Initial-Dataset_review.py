import pandas as pd

# Load dataset
data = pd.read_csv('DataSet/online_retail_II.csv')

# Basic exploration
print("Dataset Shape:", data.shape)
print("\nColumns:", data.columns)
print("\nFirst 5 rows:\n", data.head())
print("\nMissing Values:\n", data.isnull().sum())