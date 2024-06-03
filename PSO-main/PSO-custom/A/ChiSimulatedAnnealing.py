import os
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Load CSV file
data_folder = 'DATA'
input_data = input("Provide the name of the csv file: ")
file_path = os.path.join(data_folder, input_data)

data = pd.read_csv(file_path)

# Split the dataset into features and target
X = data.iloc[:, :-1]  # All columns except the last one (features)
y = data.iloc[:, -1]   # The last column (target)

# Encode target variable if it's categorical
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# Standardize features (optional but recommended for chi-squared)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform chi-squared feature selection
k = int(input("Enter the number of top features to select using chi-squared: "))
chi2_selector = SelectKBest(chi2, k=k)
X_kbest = chi2_selector.fit_transform(X_scaled, y)

# Get selected feature names
selected_features_chi2 = X.columns[chi2_selector.get_support()]

print("Chi-Squared Selected Features:\n", selected_features_chi2)

# Define the simulated annealing algorithm
class SimulatedAnnealingFeatureSelection:
    def __init__(self, estimator, X, y, initial_temp=100, final_temp=1, alpha=0.9, max_iter=1000):
        self.estimator = estimator
        self.X = X
        self.y = y
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha
        self.max_iter = max_iter
        self.n_features = X.shape[1]

    def objective_function(self, selected_features):
        X_subset = self.X[:, selected_features]
        X_train, X_test, y_train, y_test = train_test_split(X_subset, self.y, test_size=0.3, random_state=42)
        self.estimator.fit(X_train, y_train)
        y_pred = self.estimator.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def random_neighbor(self, current_features):
        neighbor = current_features.copy()
        idx = np.random.randint(0, self.n_features)
        neighbor[idx] = not neighbor[idx]
        return neighbor

    def acceptance_probability(self, old_cost, new_cost, temperature):
        if new_cost > old_cost:
            return 1.0
        return np.exp((new_cost - old_cost) / temperature)

    def fit(self):
        current_features = np.random.choice([True, False], size=self.n_features)
        current_cost = self.objective_function(current_features)
        best_features = current_features.copy()
        best_cost = current_cost

        temperature = self.initial_temp
        for iteration in range(self.max_iter):
            if temperature <= self.final_temp:
                break
            new_features = self.random_neighbor(current_features)
            new_cost = self.objective_function(new_features)
            if self.acceptance_probability(current_cost, new_cost, temperature) > np.random.rand():
                current_features = new_features
                current_cost = new_cost
                if new_cost > best_cost:
                    best_features = new_features
                    best_cost = new_cost
            temperature *= self.alpha

        return best_features

# Apply simulated annealing on the subset of features selected by chi-squared
X_kbest_scaled = StandardScaler().fit_transform(X[selected_features_chi2])
sa_selector = SimulatedAnnealingFeatureSelection(LogisticRegression(), X_kbest_scaled, y)
selected_features_sa = sa_selector.fit()

final_selected_features = selected_features_chi2[selected_features_sa]

print("Final Selected Features After Simulated Annealing:\n", final_selected_features)

# Create a new DataFrame with the selected features
X_selected_final = X[final_selected_features]
selected_data_final = pd.concat([X_selected_final, y], axis=1)

# Generate new file name
output_file_name = "New" + os.path.splitext(input_data)[0] + ".csv"
output_file_path = os.path.join(data_folder, output_file_name)
selected_data_final.to_csv(output_file_path, index=False)

print("\nSelected features have been saved to:", output_file_path)
