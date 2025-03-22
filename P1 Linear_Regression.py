# Linear Regression 
# Practical 1 : Predict disease progession using linear regression on the Diabetes dataset and evaluate model performance

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

diabetes = datasets.load_diabetes()

X = diabetes.data  # X - feature vectors
y = diabetes.target  # y - Target values

# Splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Create linear regression object
lin_reg = linear_model.LinearRegression() 

# Train the model
lin_reg.fit(X_train, y_train)

# Predict values for x_test data
predicted = lin_reg.predict(X_test)


# Regression coefficients
print('\n Coefficients are : \n', lin_reg.coef_)
# Intercept
print('\n Intercept : ', lin_reg.intercept_)
# vaiance score : 1 means perfect prediction
print('Variance score : ', lin_reg.score(X_test, y_test))

# Mean Squared Error
print("Mean squared error : %.2f\n"
      % mean_squared_error(y_test, predicted))

expected = y_test

# Plot teh graph for expected and predicted values
plt.title('Linear Regression (DIABETES Dataset)')
plt.scatter(expected, predicted, c='b', marker='.', s=40)
plt.plot(np.linspace(0, 330, 100), np.linspace(0, 330, 100), '--r', linewidth=2)

plt.show()

