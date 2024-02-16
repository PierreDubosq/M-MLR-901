import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load NPY file
data = np.load("data.npy")

def cluster_hierarchical(data, distance_threshold):
    """
    Do a hierarchical clustering.

    Args:
      data: a table containing our data.
      distance_threshold: determine the linkage distance.

    Returns:
      None
    """

    # setup the hierarchical clustering
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
    clusters = model.fit_predict(data)

    print("Clusters assignés :", clusters)
    graph_hierarchical(model, data)

def graph_hierarchical(model, data):
    """
    Draw a dendrogramme.

    Args:
      data: a table containing our data.
      model: model of the hierarchical clustering.

    Returns:
      None
    """


    # View a dendrogram to visualize the hierarchy
    plt.title('Dendrogramme hiérarchique')
    plot_dendrogram(model, data, truncate_mode='level', p=3)
    plt.xlabel("Index de l'échantillon")
    plt.ylabel("Distance euclidienne")
    plt.show()

def plot_dendrogram(model, data, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = linkage(data, method='ward')
    dendrogram(linkage_matrix, **kwargs)

def find_optimal_k(data, max_clusters=10):
    """
    Do a hierarchical clustering with a Elbow curve heuristic.

    Args:
      data: a table containing our data.
      max_clusters: max cluster create.

    Returns:
      None
    """

    # setup the hierarchical clustering
    distortions = []
    for i in range(1, max_clusters + 1):
        model = AgglomerativeClustering(n_clusters=None, distance_threshold=i)
        clusters = model.fit_predict(data)
        distortions.append(len(np.unique(clusters)))

    # Plot the Elbow curve
    plt.plot(range(1, max_clusters + 1), distortions, marker='o')
    plt.xlabel('Distance Threshold')
    plt.ylabel('Number of Clusters')
    plt.title('Elbow Method for Optimal Distance Threshold (Hierarchical)')
    plt.show()

cluster_hierarchical(data, distance_threshold=0.5)
find_optimal_k(data)