import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
import numpy as np
df = pd.read_csv("income.csv")
print("")
print("Part 1")
print("")
scaler = MinMaxScaler()
df_scaled = df.copy() 
df_scaled[['Age', 'Income($)']] = scaler.fit_transform(df[['Age', 'Income($)']])
sse = []
silhouette_scores = []
k_rng = range(2, 10)  
print("Calculating SSE and Silhouette Scores...")
for k in k_rng:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(df_scaled[['Age', 'Income($)']])
    sse.append(km.inertia_)  
    silhouette_avg = silhouette_score(df_scaled[['Age', 'Income($)']], km.labels_)
    silhouette_scores.append(silhouette_avg)
    print(f"For k = {k}, Silhouette Score: {silhouette_avg:.4f}")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_rng, sse, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of squared errors (SSE)')
plt.title('Elbow Method (on Scaled Data)')
plt.subplot(1, 2, 2)
plt.plot(k_rng, silhouette_scores, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis (on Scaled Data)')
plt.tight_layout()
plt.show()
optimal_k = 3
print(f"\nTraining final model with optimal k = {optimal_k}...")
final_km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_scaled['cluster'] = final_km.fit_predict(df_scaled[['Age', 'Income($)']])
df['cluster'] = df_scaled['cluster']
scaled_centroids = final_km.cluster_centers_
plt.figure(figsize=(8, 6))
df1 = df_scaled[df_scaled.cluster == 0]
df2 = df_scaled[df_scaled.cluster == 1]
df3 = df_scaled[df_scaled.cluster == 2]
plt.scatter(df1.Age, df1['Income($)'], color='green', label='Cluster 0')
plt.scatter(df2.Age, df2['Income($)'], color='red', label='Cluster 1')
plt.scatter(df3.Age, df3['Income($)'], color='black', label='Cluster 2')
plt.scatter(scaled_centroids[:, 0], scaled_centroids[:, 1], color='purple', marker='*', s=300, label='Centroids')
plt.title('KMeans Clusters (on Scaled Data)')
plt.xlabel('Age (Scaled)')
plt.ylabel('Income ($) (Scaled)')
plt.legend()
plt.show()
original_centroids = scaler.inverse_transform(scaled_centroids)
print("\nFinal Centroids (on original scale - Age, Income):")
print(original_centroids)

print("")
print("Part 2")
print("")


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_cluster = df[['petal length (cm)', 'petal width (cm)']]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df_cluster)
sse = []
silhouette_scores = []
k_rng = range(2, 10) 
print("Calculating SSE and Silhouette Scores for k=2 to 9...")
for k in k_rng:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    sse.append(km.inertia_)
    
    silhouette_avg = silhouette_score(X_scaled, km.labels_)
    silhouette_scores.append(silhouette_avg)
    print(f"For k = {k}, SSE: {km.inertia_:.4f}, Silhouette Score: {silhouette_avg:.4f}")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1) 
plt.plot(k_rng, sse, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of squared errors (SSE)')
plt.title('Elbow Method (on Scaled Data)')
plt.grid(True)
plt.subplot(1, 2, 2) 
plt.plot(k_rng, silhouette_scores, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis (on Scaled Data)')
plt.grid(True)
plt.tight_layout() 
plt.savefig('iris_combined_cluster_analysis.png')
plt.show()
print("Combined plot 'iris_combined_cluster_analysis.png' has been generated.")
