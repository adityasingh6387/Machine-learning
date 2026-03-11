import numpy as np
from sklearn.linear_model import LinearRegression

# Area of houses (sq ft)
area = np.array([[1000], [1500], [2000], [2500], [3000]])

# Prices of houses
price = np.array([200000, 300000, 400000, 500000, 600000])

# Create model
model = LinearRegression()

# Train model
model.fit(area, price)

# Predict price of a house with 2200 sq ft
predicted_price = model.predict([[2200]])

print("Predicted House Price:", predicted_price)
