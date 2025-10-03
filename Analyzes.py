import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_csv('DataSet\online_retail_II_v1.csv')

sales_by_country = data.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(5)
print("Top 5 Countries by Revenue:\n", sales_by_country)


sales_by_country.plot(kind='bar')
plt.title('Top 5 Countries by Revenue')
plt.xlabel('Country')
plt.ylabel('Revenue')
plt.xticks(rotation=45)
plt.show()


monthly_revenue = data.groupby('MonthYear')['Revenue'].sum()
monthly_revenue.plot(kind='line')
plt.title('Monthly Revenue Trend')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.show()


sns.histplot(data['UnitPrice'], bins=50, kde=True)
plt.title('UnitPrice Distribution')
plt.xlabel('UnitPrice')
plt.ylabel('Frequency')
plt.show()