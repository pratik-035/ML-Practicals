# Practical 3 :  Decsion Tree Classifer
# Build a decision tree model on a weather dataset to determine if a tennis game hsould be played based oon conditions.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Data preparation 

data = {
    'Outlook' : ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 
                 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'],
    'Temperature' : ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild',
                     'Mild', 'Hot', 'Mild'],
    'Humidity' : ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal',
                  'Normal', 'High', 'Normal', 'High'],
    'Wind' : ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 
              'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis' : ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes',
                    'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

df_encoded = df.apply(lambda x: pd.factorize(x)[0])

X = df_encoded.drop('PlayTennis', axis=1)
y = df_encoded['PlayTennis']

# Build decision tree using sklearn
clf = DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, y)

# Plot the decision Tree
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, 
               feature_names=df.columns[:-1].tolist(), # Convert feature names to list
               class_names=['No', 'Yes'],
               filled=True,
               rounded=True,
               fontsize=10)
plt.show()

# Function to predict the outcome using the decision tree

def predict(query, clf, feature_names):
    query_encoded = [pd.factorize(df[feature])[0][df[feature] == 
                    query[feature]].tolist()[0] for feature in feature_names]
    prediction = clf.predict([query_encoded])
    return 'Yes' if prediction == 1 else 'No'

# Sample query 
query = {'Outlook' : 'Sunny', 'Temperature' : 'Cool', 'Humidity' : 'High', 'Wind' : 'Strong'}
prediction = predict(query, clf, df.columns[:-1].tolist())
print(f"Prediction for {query} : {prediction}")

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = df.drop('PlayTennis', axis=1)
y = df['PlayTennis']

# Use OneHotEncoder to encode categorical features
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy}")

# sampole input for prediction 

sample_input = {
    'Outlook' : ['Sunny'],
    'Temperature' : ['Cool'],
    'Humidty' : ['High'],
    'Wind' : ['Weak']
}


# Encode the sample input using the same encoder

sample_input_encoded = encoder.transform(pd.DataFrame(sample_input))

# Make Predictions
sample_predictions = clf.predict(sample_input_encoded)

if (sample_predictions[0] == 'No'):
    print("Predictions : No, don't play tennis")
else :
    print("Predctions : Yes, play tennis")