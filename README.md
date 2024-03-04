# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load customer data
customer_data = pd.read_csv('customer_data.csv')

# Data preprocessing
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data)

# Determine optimal number of clusters
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_data)
    silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))

optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

# Perform K-means clustering
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(scaled_data)

# Add cluster labels to the customer data
customer_data['Cluster'] = kmeans.labels_

# Calculate customer lifetime value
customer_data['Customer Lifetime Value'] = customer_data['Average Purchase Value'] * customer_data['Purchase Frequency'] * customer_data['Customer Lifespan']

# Save clustered data to a file
customer_data.to_csv('clustered_customer_data.csv', index=False)
