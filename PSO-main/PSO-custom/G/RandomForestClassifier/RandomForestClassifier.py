"""
Author: [Your Name]
Date: [Date]
Description: This script performs classification on the TCGA dataset using a Random Forest Classifier and cross-validation.

- The script loads the TCGA dataset, removes unnecessary columns, and separates features and target.
- It trains a Random Forest Classifier on the full dataset to determine feature importance.
- The top features are selected based on importance scores to train the final Random Forest Classifier.
- The script evaluates the model using accuracy and cross-validation scores.
- Results are saved to a specified CSV file.

Steps:
1. Load the dataset and clean it by removing unnamed columns.
2. Train a Random Forest Classifier on the full dataset to determine feature importance.
3. Select top features based on importance scores.
4. Separate the features (X) and target variable (Y).
5. Encode any categorical features and the target variable.
6. Split the data into training and testing sets with stratification.
7. Standardize the feature data.
8. Initialize and train a Random Forest Classifier.
9. Evaluate the classifier using accuracy and cross-validation scores.
10. Save the results to a CSV file.

Required Libraries:
- pandas
- numpy
- scikit-learn
- matplotlib (if needed for further analysis)

Usage:
1. Ensure the necessary libraries are installed.
2. Set the correct file paths for input dataset and output results CSV.
3. Run the script to perform classification and save the results.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Load dataset
file_input = input("Please enter file input path: ")
data = pd.read_csv(file_input)

# Drop unnamed columns
data = data.iloc[:, ~data.columns.str.contains('^Unnamed')]

# Separate features (X) and target variable (Y)
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

# Convert any categorical features in X to numeric using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Encode the target variable
le = LabelEncoder()
Y = le.fit_transform(Y)

# Split data into training and testing sets with stratification
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, stratify=Y, test_size=0.3, random_state=42)

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the training data, transform the testing data
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

# Train a Random Forest Classifier to determine feature importance
rf = RandomForestClassifier(random_state=42)
rf.fit(xtrain, ytrain)

# Get feature importances
importances = rf.feature_importances_

# Create a DataFrame for feature importances
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

# Sort the DataFrame by importance
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Select top N features based on importance (e.g., top 10 features)
top_n_features = feature_importances['Feature'].head(10).tolist()

# Use only the top N features for training the final model
X_top_n = X[top_n_features]

# Split data into training and testing sets with stratification using top N features
xtrain_top_n, xtest_top_n, ytrain_top_n, ytest_top_n = train_test_split(X_top_n, Y, stratify=Y, test_size=0.3, random_state=42)

# Fit and transform the training data, transform the testing data using top N features
xtrain_top_n = scaler.fit_transform(xtrain_top_n)
xtest_top_n = scaler.transform(xtest_top_n)

# Train the final Random Forest classifier with top N features
rf_final = RandomForestClassifier(random_state=42)
rf_final.fit(xtrain_top_n, ytrain_top_n)

# Make predictions
ypred = rf_final.predict(xtest_top_n)

# Calculate accuracy for Random Forest classifier
accuracy = accuracy_score(ytest_top_n, ypred)

# Print the accuracy
print("Accuracy of Random Forest Classifier:", accuracy)

# Cross-validation for a more robust performance estimate
cv_scores = cross_val_score(rf_final, X_top_n, Y, cv=5)
print(f"Cross-validated accuracy scores: {cv_scores}")
print(f"Mean cross-validated accuracy: {np.mean(cv_scores)}")

# Save results to a CSV file
results = pd.DataFrame({
    "Model": ["Random Forest"],
    "Accuracy": [accuracy],
    "Mean Cross-Validated Accuracy": [np.mean(cv_scores)]
})
output_path_input = input("Please enter output csv file name: ")
results.to_csv('PSO-main/PSO-custom/G/RandomForestClassifier/results/' + output_path_input + '.csv', index=False)
