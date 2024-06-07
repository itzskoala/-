import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

def load_and_prepare_data(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Convert non-numeric columns to numeric where possible
    for col in df.columns[:-1]:  # All columns except the last one
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN values (which were non-numeric entries that couldn't be converted)
    df.dropna(inplace=True)
    
    return df

def select_features(X, y):
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize the model
    model = LogisticRegression(max_iter=5000)
    
    # Determine the optimal number of features
    best_score = 0
    best_num_features = 1
    for num_features in range(1, X.shape[1] + 1):
        rfe = RFE(model, n_features_to_select=num_features)
        rfe.fit(X_scaled, y)
        score = np.mean(cross_val_score(rfe, X_scaled, y, cv=5, scoring='accuracy'))
        if score > best_score:
            best_score = score
            best_num_features = num_features
    
    # Perform RFE with the optimal number of features
    rfe = RFE(model, n_features_to_select=best_num_features)
    rfe.fit(X_scaled, y)
    
    # Get the ranking of the features
    selected_features = X.columns[rfe.support_]
    
    return selected_features

def save_selected_features(df, selected_features, output_file_path):
    # Keep only the selected features
    df_selected = df[selected_features.tolist() + [df.columns[-1]]]  # Add the target column back
    df_selected.to_csv(output_file_path, index=False)
    print(f"Data with selected features has been saved to {output_file_path}")

def main():
    # Get dataset file path from user
    csv_file_path = input("Enter the path to your CSV file: ")
    
    # Load and prepare the data
    df = load_and_prepare_data(csv_file_path)
    
    # Separate features and target
    X = df.iloc[:, :-1]  # All columns except the last one
    y = df.iloc[:, -1]   # The last column
    
    # Select features

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    
    selected_features = select_features(X, y)
    
    # Save the data with selected features
    output_file_path = 'RFE_FS.csv'
    save_selected_features(df, selected_features, output_file_path)

if __name__ == "__main__":
    main()
