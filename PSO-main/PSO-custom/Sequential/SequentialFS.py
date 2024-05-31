import os
import sys
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

selection_direction = input("Do you want to perform forward selection or backward selection?\nFor forward selection, type 1\nFor backward selection, type 2\n")

if selection_direction == "1":
    selector = SequentialFeatureSelector(logreg, n_features_to_select=5, scoring = 'accuracy')
elif selection_direction == "2":
    selector = SequentialFeatureSelector(logreg, n_features_to_select=5, direction = 'backward', scoring = 'accuracy')
else:
    sys.exit()

selector.fit(X, y)

selected_features = selector.get_support()

print("Selected features: ", list(X.columns[selected_features]))