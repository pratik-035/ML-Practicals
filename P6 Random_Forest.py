# Practical 6 : Random Forest 
# Use a random forest model to classify Iiris species, analyze performance, and visualize decision boundaries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the dataset 
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names).iloc[:, :2] # Only the first two features
y = pd.DataFrame(iris.target, columns=['species'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# function to plot the decision boundary
def plot_decision_boundary(clf, X, y, title):
    
    # create meshgrid
    x_min, x_max = X.iloc[:, 0].min() - 1 , X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1 , X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Predict for the entire grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the contour and training points
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y.values.ravel(), s=40, 
                edgecolors='k', cmap=plt.cm.RdYlBu)
    plt.title(title)
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.show()


# Decison Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Make Prediction with Decision Tree
dt_predictions = dt_model.predict(X_test)

# Decision Tree Accuracy and confusion matrix
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_confusion_matrix = confusion_matrix(y_test, dt_predictions)

print(f"Decision Tree Accuracy : {dt_accuracy}")
print(f"Classification Report : ")
print(classification_report(y_test, dt_predictions))

# Plot the confusionj matrix for decision tree
sns.heatmap(dt_confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Decison Tree Confusion Matrix')
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.show()

# plot the decison boundary for Decision Tree
plot_decision_boundary(dt_model, X_test, y_test, "Decision Tree Decision Boundary")

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train.values.ravel())

# Make predictions with Random Forest
rf_predictions = rf_model.predict(X_test)

# Random Forest Accuracy and confusion matrix
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_confusion_matrix = confusion_matrix(y_test, rf_predictions)

print(f"Random Forest Accuracy : {rf_accuracy}")
print(f"Random Forest Classification Report : ")
print(classification_report(y_test, rf_predictions))

# Plot the confusion matrix for Random Forest
sns.heatmap(rf_confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
plt.show()

# Plot the Decision Boundary For Random Forest
plot_decision_boundary(rf_model, X_test, y_test, "Random Forest Decision Boundary")

