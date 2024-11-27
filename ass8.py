import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as shc
data = pd.read_csv(r'F:\sem5\DWDM\lab\healthcare_noshows.csv')
features = ['Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']
X = data[features].head(200).fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
plt.figure(figsize=(10, 7))
plt.title("Dendrogram for Healthcare No-shows Data")
dend = shc.dendrogram(shc.linkage(X_scaled, method='ward'))
plt.show()
optimal_clusters = 7
agg_cluster = AgglomerativeClustering(n_clusters=optimal_clusters, linkage='ward')
y_clusters = agg_cluster.fit_predict(X_scaled)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_clusters, cmap='rainbow', marker='o', edgecolor='k')
plt.title(f"Agglomerative Clustering ({optimal_clusters} clusters)")
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster Label')
plt.grid()
plt.show()