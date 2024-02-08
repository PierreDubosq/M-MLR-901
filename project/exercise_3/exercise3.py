import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# Load NPY file
data = np.load('./data.npy')

# Print all the content
print(data)

#for i, n in enumerate(range(4)):
#    nth_elements = data[:, n]
#    
#    # Create a subplot for each index
#    plt.subplot(len(range(4)), 1, i + 1)
#    plt.plot(nth_elements)
#    plt.xlabel('Index of Sub-array')
#    plt.ylabel(f'Value at Index {n}')
#    plt.title(f'Graph of the {n}th Element in Each Sub-array')
#
#plt.tight_layout()
#plt.show()

kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

# Step 5: Get Cluster Assignments
cluster_assignments = kmeans.predict(data)

# Step 6: Visualize the Results (Optional)
# If your data is in 4D, reduce dimensionality for visualization
# For example, using PCA for 2D visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(data)

# Plot clustered data
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_assignments, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering of 4D Data')
plt.colorbar(label='Cluster')
plt.show()