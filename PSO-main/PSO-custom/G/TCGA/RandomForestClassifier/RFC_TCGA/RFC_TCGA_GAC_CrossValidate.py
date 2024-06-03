"""
Author: [Your Name]
Date: [Date]
Description: This script performs classification on the TCGA dataset using a Random Forest Classifier and cross-validation.

- The script loads the TCGA dataset, removes unnecessary columns, and separates features and target.
- It loads the selected features from a previously saved CSV file.
- The selected features are used to train a Random Forest Classifier.
- The script evaluates the model using accuracy and cross-validation scores.
- Results are saved to a specified CSV file.

Steps:
1. Load the TCGA dataset and clean it by removing unnamed columns.
2. Load the selected features from a CSV file.
3. Ensure the selected features are present in the dataset.
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
2. Set the correct file paths for input dataset, selected features CSV, and output results CSV.
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
data = pd.read_csv('C:/Users/guill/Downloads/TCGA.csv')

# Drop unnamed columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Load selected features from the feature selection method
selected_features_df = pd.read_csv('PSO-main/PSO-custom/G/TCGA/GeneticAlgorithmClassifier/results/features_TCGA_GA.csv')
selected_features = selected_features_df.columns.tolist()

# Ensure the selected features are present in the data
selected_features = [feature for feature in selected_features if feature in data.columns]

# Check if selected features list is not empty
if not selected_features:
    raise ValueError("No valid features found in the selected features list.")

# Separate features (X) and target variable (Y) using selected features
X = data[selected_features]

# Convert any categorical features in X to numeric using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Encode the target variable
le = LabelEncoder()
Y = le.fit_transform(data['Class'])

# Split data into training and testing sets with stratification
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, stratify=Y, test_size=0.3, random_state=42)

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

# Cross-validation for a more robust performance estimate
cv_scores = cross_val_score(rf, X, Y, cv=5)
print(f"Cross-validated accuracy scores: {cv_scores}")
print(f"Mean cross-validated accuracy: {np.mean(cv_scores)}")

# Save results to a CSV file
results = pd.DataFrame({
    "Model": ["Random Forest"],
    "Accuracy": [accuracy],
    "Mean Cross-Validated Accuracy": [np.mean(cv_scores)]
})
results.to_csv('PSO-main/PSO-custom/G/TCGA/RandomForestClassifier/results/RFC_TCGA_GA_accuracies.csv', index=False)
