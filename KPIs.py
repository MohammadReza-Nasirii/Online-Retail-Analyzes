import pandas as pd

# Load dataset
data = pd.read_csv('DataSet/online_retail_II_v1.csv')

# Ensure numeric columns
data['Quantity'] = pd.to_numeric(data['Quantity'], errors='coerce')
data['Price'] = pd.to_numeric(data['Price'], errors='coerce')

# Calculate Total Revenue
data['Revenue'] = data['Quantity'] * data['Price']
total_revenue = data['Revenue'].sum()
print(f"Total Revenue: {total_revenue}")

# Calculate Average Order Value
orders = data.groupby('Invoice')['Revenue'].sum()
aov = orders.mean()
print(f"Average Order Value (AOV): {aov}")

# Calculate Customer Retention Rate
unique_customers = data['Customer ID'].nunique()
returning_customers = data.groupby('Customer ID')['Invoice'].nunique()
retention_rate = (returning_customers[returning_customers > 1].count() / unique_customers) * 100
print(f"Customer Retention Rate: {retention_rate}%")

# Calculate Sales Growth
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], errors='coerce')
data['MonthYear'] = data['InvoiceDate'].dt.to_period('M')
monthly_sales = data.groupby('MonthYear')['Revenue'].sum()
sales_growth = monthly_sales.pct_change().mean() * 100
print(f"Average Monthly Sales Growth: {sales_growth}%")

# Calculate Top Products by Sales (using StockCode instead of Description)
top_products = data.groupby('StockCode')['Revenue'].sum().sort_values(ascending=False).head(5)
print("Top 5 Products by Sales (StockCode):\n", top_products)

# Save KPIs to file (optional)
kpis = pd.DataFrame({
    'KPI': ['Total Revenue', 'AOV', 'Retention Rate', 'Sales Growth', 'Top Products'],
    'Value': [total_revenue, aov, retention_rate, sales_growth, top_products.to_string()]
})
kpis.to_csv('DataSet/kpi_summary.csv', index=False)