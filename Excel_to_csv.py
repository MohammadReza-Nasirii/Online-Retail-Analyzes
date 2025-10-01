import pandas as pd

# Load the Excel file
excel_file = 'DataSet/online_retail_II.xlsx'
df = pd.read_excel(excel_file)

# Save as CSV
csv_file = 'DataSet/online_retail_II.csv'
df.to_csv(csv_file, index=False)

print(f"Converted {excel_file} to {csv_file} successfully!")