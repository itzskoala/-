import os
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

data_folder = 'DATA'
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

# Optionally, select top N features based on mutual information scores
top_n = int(input("Enter the number of top features to select: "))
top_features = mi_scores_sorted.head(top_n).index

print("\nTop {} Features:\n".format(top_n), list(top_features))


# Create a new DataFrame with the selected features
X_selected = X[top_features]
selected_data = pd.concat([X_selected, y], axis=1)

# Save the selected features to a new CSV file
output_file_name = input("Provide the name of the output csv file: ")
output_file_path = os.path.join(data_folder, output_file_name)
selected_data.to_csv(output_file_path, index=False)

print("\nSelected features have been saved to:", output_file_path)