import os
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

data_folder = 'PSO-main/data'
input_data = input("Provide the name of the csv file: ")
file_path = os.path.join(data_folder, input_data)

data = pd.read_csv(file_path)

# Split the dataset into features and target
X = data.iloc[:, :-1]  # All columns except the last one (features)
y = data.iloc[:, -1]   # The last column (target)

# Perform mutual information feature selection
mi_scores = mutual_info_classif(X, y)

# Convert to a pandas Series for easy handling
mi_scores_series = pd.Series(mi_scores, index=X.columns)

# Sort the scores in descending order
mi_scores_sorted = mi_scores_series.sort_values(ascending=False)

print("Mutual Information Scores:\n", mi_scores_sorted)

# Automatically select features based on a threshold (e.g., mean score)
threshold = mi_scores_sorted.mean()
selected_features = mi_scores_sorted[mi_scores_sorted > threshold].index

print("\nSelected Features Based on Threshold:\n", list(selected_features))

# Create a new DataFrame with the selected features
X_selected = X[selected_features]
selected_data = pd.concat([X_selected, y], axis=1)

# Save the selected features to a new CSV file
output_file_name = "New" + os.path.splitext(input_data)[0] + ".csv"
output_file_path = os.path.join(data_folder, output_file_name)
selected_data.to_csv(output_file_path, index=False)

print("\nSelected features have been saved to:", output_file_path)

# Optionally, evaluate model performance using selected features
rf = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(rf, X_selected, y, cv=5, scoring='accuracy')

print(f"\nModel Accuracy with Selected Features: {scores.mean():.4f} ± {scores.std():.4f}")
