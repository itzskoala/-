# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('C:/Users/guill/Downloads/TCGA.csv')

# Drop unnamed columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Separate features (X) and target variable (Y)
X = data.drop("Class", axis=1)
Y = data.Class

# Split data into training and testing sets with stratification
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, stratify=Y, test_size=0.3)

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the training data, transform the testing data
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

# Initialize Decision Tree and Random Forest classifiers
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

# Train the classifiers
dt.fit(xtrain, ytrain)
rf.fit(xtrain, ytrain)

# Make predictions
ypred1 = dt.predict(xtest)
ypred2 = rf.predict(xtest)

# Calculate accuracy for Decision Tree and Random Forest classifiers
a = accuracy_score(ytest, ypred1)
b = accuracy_score(ytest, ypred2)

# Print the accuracies
print("Accuracy of Decision Tree Classifier:", a)
print("Accuracy of Random Forest Classifier:", b)
