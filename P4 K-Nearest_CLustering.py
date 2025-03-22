# k-Nearest Neighbor 
# Practical 4 : Classify Iris species using K-NN, visualize decision boundaries and experiment with different K-Values.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

iris = load_iris()

X = iris.data[:, :2] # Select only the first two features (sepal length and sepal width)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k = 3
knn_classifier = KNeighborsClassifier(n_neighbors=k)

knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)

# Evaluate Classifiers performance
print("Confusion Matrix :\n")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report : \n")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(7, 4))

# Plot the training data points
plt.scatter(
    X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', label="Training Data"
)

# Plot the testing data points
plt.scatter(
    X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='x', s=100, label='Testing Data'
)

# Plot the decision boundaries

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1 , X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

Z = knn_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.5, levels=range(4))
plt.colorbar()

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title(f'K-NN Classifier (k={k}) on Iris Dataset (2 Features)')
plt.legend()
plt.show()
