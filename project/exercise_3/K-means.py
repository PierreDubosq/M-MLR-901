import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

# Load NPY file
data = np.load('c:/Users/theor/Documents/GitHub/M-MLR-901/project/exercise_3/data.npy')

def find_optimal_k(data, max_clusters=10):
    """
    Do a K-Means clustering with a Elbow curve heuristic.

    Args:
      data: a table containing our data.
      max_clusters: max cluster create.

    Returns:
      None
    """

    # setup the Kmeans Clustering
    distortions = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)

    # Plot the Elbow curve
    plt.plot(range(1, max_clusters + 1), distortions, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal K')
    plt.show()

def plot_dendrogram(data):
    """
    Do a K-Means clustering with a dendrogram heuristic.

    Args:
      data: a table containing our data.

    Returns:
      None
    """

    # setup the Kmeans Clustering
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_

    # Perform hierarchical clustering on the centroids
    linkage_matrix = linkage(centroids, method='ward')

    # Plot the dendrogram
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix, labels=[f'Cluster {i+1}' for i in range(len(centroids))])
    plt.title('Dendrogram for K-means Centroids')
    plt.xlabel('Clusters')
    plt.ylabel('Distance')
    plt.show()

find_optimal_k(data)
plot_dendrogram(data)

