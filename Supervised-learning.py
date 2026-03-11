import numpy as np
from sklearn.linear_model import LinearRegression

# Training Data (Hours studied vs Marks)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([35, 40, 50, 55, 65])

# Create Model
model = LinearRegression()

# Train Model
model.fit(X, y)

# Predict Marks for 6 hours study
prediction = model.predict([[6]])

print("Predicted Marks:", prediction)
