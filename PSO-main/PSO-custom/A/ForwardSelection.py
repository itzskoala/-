import os
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression


data_folder = 'DATA'
input_data = input("Provide the name of the csv file: ")
file_path = os.path.join(data_folder, input_data)

data = pd.read_csv(file_path)

# Split the dataset into features and target
X = data.iloc[:, :-1]  # All columns except the last one (features)
y = data.iloc[:, -1]   # The last column (target)

logreg = LogisticRegression()

selector = SequentialFeatureSelector(logreg, n_features_to_select=5, scoring = 'accuracy')

selector.fit(X, y)

selected_features = selector.get_support()

print("Selected features: ", list(X.columns[selected_features]))

# Create a new DataFrame with the selected features
X_selected = X.loc[:, selected_features]
selected_data = pd.concat([X_selected, y], axis=1)

# Save the selected features to a new CSV file
output_file_name = input("Provide the name of the output csv file: ")
output_file_path = os.path.join(data_folder, output_file_name)
selected_data.to_csv(output_file_path, index=False)

print("\nSelected features have been saved to:", output_file_path)