import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def load_dataset(url):
    """
    Load the Wine dataset from the given URL.

    Parameters:
    url (str): The URL of the dataset.

    Returns:
    pandas.DataFrame: The loaded Wine dataset.
    """
    column_names = ["Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", 
                    "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", 
                    "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]

    wine_data = pd.read_csv(url, names=column_names)
    return wine_data

def explore_dataset(wine_data):
    """
    Perform initial exploration of the dataset.

    Parameters:
    - wine_data: pandas DataFrame
        The dataset to be explored.

    Returns:
    None
    """
    print(wine_data.head())
    print(wine_data.describe())
    print(wine_data.isnull().sum())

    corr_matrix = wine_data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()

def preprocess_data(X):
    """
    Preprocess the data by scaling the features.

    Parameters:
    X (array-like): The input data.

    Returns:
    array-like: The preprocessed data with scaled features.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def perform_unsupervised_learning(X_scaled):
    """
    Apply unsupervised learning techniques: KMeans clustering and PCA.

    Parameters:
    - X_scaled: numpy array
        The scaled input data.

    Returns:
    - labels: numpy array
        The cluster labels assigned by KMeans.
    - X_pca: numpy array
        The reduced-dimensional data obtained through PCA.
    - pca: PCA object
        The PCA object used for dimensionality reduction.
    """
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    return kmeans.labels_, X_pca, pca

def visualize_results(X_pca, labels):
    """
    Visualize the results of unsupervised learning.

    Parameters:
    - X_pca (array-like): The PCA-transformed data.
    - labels (array-like): The cluster labels assigned to each data point.

    Returns:
    None
    """
    pca_df = pd.DataFrame(data=X_pca, columns=["PC1", "PC2"])
    pca_df["Cluster"] = labels

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="PC1", y="PC2", hue="Cluster", data=pca_df, palette="viridis")
    plt.title("PCA with Clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

def evaluate_results(X_scaled, labels, pca):
    """
    Evaluate the results of unsupervised learning.

    Parameters:
    - X_scaled: The scaled feature matrix.
    - labels: The predicted labels.
    - pca: The fitted PCA model.

    Returns:
    None
    """
    silhouette_avg = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score: {silhouette_avg}")

    explained_variance_ratio = pca.explained_variance_ratio_
    print(f"Explained Variance Ratio (PC1, PC2): {explained_variance_ratio}")

def main():
    """
    Main function that loads the dataset, explores it, preprocesses the data,
    performs unsupervised learning, visualizes the results, and evaluates the results.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    wine_data = load_dataset(url)

    explore_dataset(wine_data)

    X = wine_data.drop("Class", axis=1)

    X_scaled = preprocess_data(X)

    labels, X_pca, pca = perform_unsupervised_learning(X_scaled)

    visualize_results(X_pca, labels)

    evaluate_results(X_scaled, labels, pca)


if __name__ == "__main__":
    main()