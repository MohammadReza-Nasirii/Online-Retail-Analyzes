import matplotlib
matplotlib.use('TkAgg')  # Fix for PyCharm Matplotlib backend
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Load dataset
data = pd.read_csv('DataSet/online_retail_II_v1.csv')

# Prepare data for regression
data_model = data[(data['Quantity'] > 0) & (data['Price'] > 0)]  # Filter positive values
data_model['InvoiceDate'] = pd.to_datetime(data_model['InvoiceDate'], errors='coerce')
data_model['Month'] = data_model['InvoiceDate'].dt.month  # Add month as a feature

# Define features and target
X = data_model[['Price', 'Month']]  # Multiple features: Price and Month
y = data_model['Quantity']

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model 1: Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)
print(f"Linear Regression - MSE: {mse_lin}, R²: {r2_lin}")

# Model 2: Polynomial Regression (degree 3)
poly = PolynomialFeatures(degree=3)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)
y_pred_poly = poly_model.predict(X_poly_test)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)
print(f"Polynomial Regression (Degree 3) - MSE: {mse_poly}, R²: {r2_poly}")

# Comparison
print(f"Comparison: Linear R² = {r2_lin}, Polynomial R² = {r2_poly}")

# Visualization
plt.figure(figsize=(10, 5))

# Scatter plot of actual vs predicted
plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], y_test, color='blue', label='Actual')  # Use first feature (Price) for x-axis
plt.plot(np.sort(X_test[:, 0]), np.sort(y_pred_lin), color='red', label='Linear')
plt.plot(np.sort(X_test[:, 0]), np.sort(y_pred_poly), color='green', label='Polynomial')
plt.title('Comparison: Linear vs Polynomial Regression')
plt.xlabel('Price (Normalized)')
plt.ylabel('Quantity')
plt.legend()

# Residual Plot
plt.subplot(1, 2, 2)
residuals_lin = y_test - y_pred_lin
plt.scatter(X_test[:, 0], residuals_lin, color='red', label='Linear Residuals')
residuals_poly = y_test - y_pred_poly
plt.scatter(X_test[:, 0], residuals_poly, color='green', label='Polynomial Residuals')
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Price (Normalized)')
plt.ylabel('Residuals')
plt.legend()

plt.tight_layout()
plt.show()

# Save model and results
import joblib
joblib.dump(lin_model, 'models/linear_regression_model.pkl')
joblib.dump(poly_model, 'models/polynomial_regression_model.pkl')
results = pd.DataFrame({'Actual': y_test, 'Predicted_Linear': y_pred_lin, 'Predicted_Poly': y_pred_poly})
results.to_csv('data/regression_results.csv', index=False)