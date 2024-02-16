import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load NPY file
data = np.load('./data.npy')
labels = np.load('./labels.npy')

indices_to_plot = range(6)

# Print all the content
# print(labels)

#fig, axs = plt.subplots(len(indices_to_plot), 1, figsize=(8, 2 * len(indices_to_plot)))

# Plot for each index
# for i, n in enumerate(indices_to_plot):
#    nth_elements = data[:, n]
#
#    # Create a subplot for each index
#    plt.subplot(len(indices_to_plot), 1, i + 1)
#    plt.plot(nth_elements)
#    plt.xlabel('Index of Sub-array')
#    plt.ylabel(f'Value at Index {n}')
#    plt.title(f'Graph of the {n}th Element in Each Sub-array')
#
# plt.tight_layout()
# plt.show()

selected_columns = data[:, [0, 1, 5]].copy()

n_components = 2  # You can adjust this value based on your desired number of components
pca = PCA(n_components=n_components)
reduced_data = pca.fit_transform(selected_columns)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(selected_columns[:, 0], selected_columns[:, 1],
           selected_columns[:, 2], c=labels, cmap='viridis')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
plt.title('3D Scatter Plot of Manually Reduced Data with Boolean Labels')
plt.show()
# 'reduced_data' now contains the reduced-dimensional representation of the input data

# Print the explained variance ratio to see how much information is retained
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

# Afficher les résultats de la PCA en 2D avec une ellipse de confiance
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('PCA Plot of Reduced Data with Confidence Ellipse')

# Calculer les demi-axes de l'ellipse de confiance
eigenvalues = pca.explained_variance_
angle = np.arccos(pca.components_[0, 0])  # Angle de rotation
width, height = 2 * np.sqrt(eigenvalues)  # Largeur et hauteur de l'ellipse

# Dessiner l'ellipse de confiance
ellipse = plt.matplotlib.patches.Ellipse(xy=(
    0, 0), width=width, height=height, angle=np.degrees(angle), fill=False, color='b')
plt.gca().add_patch(ellipse)

plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    reduced_data, labels, test_size=0.2, random_state=42)

# Train a simple classifier (Logistic Regression for example)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Evaluate the classifier's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Classifier Accuracy: {accuracy:.2f}")