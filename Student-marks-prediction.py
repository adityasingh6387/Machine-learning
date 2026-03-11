import numpy as np
from sklearn.linear_model import LinearRegression

# Study hours data
hours = np.array([[1], [2], [3], [4], [5]])

# Marks obtained
marks = np.array([20, 40, 50, 70, 90])

# Create model
model = LinearRegression()

# Train model
model.fit(hours, marks)

# Predict marks for 6 hours study
predicted_marks = model.predict([[6]])

print("Predicted Marks:", predicted_marks)
