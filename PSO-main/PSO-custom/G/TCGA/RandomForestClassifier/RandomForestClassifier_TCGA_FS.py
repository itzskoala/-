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

# Load selected features from the feature selection method
selected_features = pd.read_csv('PSO-main/PSO-custom/G/TCGA/GeneticAlgorithmClassifier/results/features_TCGA_GA.csv').columns.tolist()

# Separate features (X) and target variable (Y) using selected features
X = data[selected_features]
Y = data['Class']

# Split data into training and testing sets with stratification
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, stratify=Y, test_size=0.3)

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the training data, transform the testing data
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

# Initialize Random Forest classifier
rf = RandomForestClassifier()

# Train the classifier
rf.fit(xtrain, ytrain)

# Make predictions
ypred = rf.predict(xtest)

# Calculate accuracy for Random Forest classifier
accuracy = accuracy_score(ytest, ypred)

# Print the accuracy
print("Accuracy of Random Forest Classifier:", accuracy)

# Save results to a CSV file
results = pd.DataFrame({
    "Model": ["Random Forest"],
    "Accuracy": [accuracy]
})
results.to_csv('PSO-main/PSO-custom/G/TCGA/RandomForestClassifier/results/RFC_TCGA_GA_accuracies.csv', index=False)
