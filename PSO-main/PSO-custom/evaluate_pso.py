import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_data(file_path):
    """Load data from CSV file into pandas DataFrame."""
    data = pd.read_csv(file_path)
    return data

def greedy_stepwise_feature_selection(X, y, estimator, cv=5):
    """Greedy stepwise feature selection."""
    selected_features = []
    best_score = 0
    
    while True:
        candidate_features = []
        for feature in X.columns:
            if feature not in selected_features:
                candidate_features.append(selected_features + [feature])
        
        if not candidate_features:
            break
        
        best_candidate = None
        best_candidate_score = 0
        for candidate in candidate_features:
            scores = cross_val_score(estimator, X[candidate], y, cv=cv)
            candidate_score = np.mean(scores)
            if candidate_score > best_candidate_score:
                best_candidate_score = candidate_score
                best_candidate = candidate
        
        if best_candidate_score > best_score:
            best_score = best_candidate_score
            selected_features = best_candidate
        else:
            break
    
    return selected_features, best_score

if __name__ == "__main__":
    # Path to the CSV file
    csv_file_path = input("Enter the path to the CSV file: ")
    
    # Load data
    data = load_data(csv_file_path)
    
    # Determine target column
    target_column = data.columns[-1]  # Assuming the last column is the target variable
    
    # Split features and target variable
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Choose estimator
    estimator = LogisticRegression()
    
    # Perform greedy stepwise feature selection
    selected_features, best_score = greedy_stepwise_feature_selection(X, y, estimator)
    
    print("Best selected features:", selected_features)
    print("Best cross-validation score:", best_score)
