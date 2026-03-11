import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample Data
X = np.array([[1,2], [1,4], [1,0],
              [10,2], [10,4], [10,0]])

# Create Model (2 clusters)
kmeans = KMeans(n_clusters=2)

# Train Model
kmeans.fit(X)

# Predict cluster for each data point
labels = kmeans.predict(X)

print("Cluster Labels:", labels)
print("Cluster Centers:", kmeans.cluster_centers_)
