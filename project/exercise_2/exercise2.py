import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#load data files
def readFiles():
    data = np.load('./data.npy')
    labels = np.load('./labels.npy')
    return (data, labels)

# make a graph with each column to examine which are dependings on each other
def analyzeEachColumn(data):
    indices_to_plot = data.length
    for i, n in enumerate(indices_to_plot):
         nth_elements = data[:, n]
         plt.subplot(len(indices_to_plot), 1, i + 1)
         plt.plot(nth_elements)
         plt.xlabel('Index of Sub-array')
         plt.ylabel(f'Value at Index {n}')
         plt.title(f'Graph of the {n}th Element in Each Sub-array')
    plt.tight_layout()
    plt.show()
    # => drawn conclusions based on manual verification of graphs: 3rd, 4th and 5th columns
    # are very similar. 1st is like 6th but with more amplitude
    # => to simplify the data set, we can manually remove columns 3, 4 and 6

# apply the filtering explained above
def filterColumns(data):
    return data[:, [0, 1, 5]].copy()

def draw3Ddiagram(data, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1],
               data[:, 2], c=labels, cmap='viridis')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    plt.title('3D Scatter Plot of Manually Reduced Data with Boolean Labels')
    plt.show()

def draw2Ddiagram(data, labels, pca):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('PCA Plot of Reduced Data with Confidence Ellipse')

    # calculate ellipsis
    eigenvalues = pca.explained_variance_
    angle = np.arccos(pca.components_[0, 0])  # rotation angle
    width, height = 2 * np.sqrt(eigenvalues)  # Width and height 

    # draw the ellipsis
    ellipse = plt.matplotlib.patches.Ellipse(xy=(
        0, 0), width=width, height=height, angle=np.degrees(angle), fill=False, color='b')
    plt.gca().add_patch(ellipse)

    plt.show()

def train(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Train a simple classifier (Logistic Regression for example)
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # Predict on the test set
    y_pred = classifier.predict(X_test)

    # Evaluate the classifier's performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classifier Accuracy: {accuracy:.2f}")

def main():
    #load datas
    data, labels = readFiles()

    # manually remove redundant informations
    selected_columns = filterColumns(data)

    # draw 3D diagrams of current dataset
    draw3Ddiagram(selected_columns, labels)

    #further reducing dimensions with sklearn PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(selected_columns)

    # Print the explained variance ratio to see how much information is retained
    print("Explained Variance Ratio:")
    print(pca.explained_variance_ratio_)

    # Display PCA resultats in 2D with a confidence ellipsis
    draw2Ddiagram(reduced_data, labels, pca)

    # Split the data into training and testing sets
    train(reduced_data, labels)

main()