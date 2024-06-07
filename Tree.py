import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler


def feature_selection():
    # Get user inputs for file paths
    input_csv = input("Enter the path to the input CSV file: ")
    output_csv = 'Tree_FS.csv'
    
    # Read the dataset
    data = pd.read_csv(input_csv)
    
    # Check if the last column is numeric or categorical (target)
    if pd.api.types.is_numeric_dtype(data.iloc[:, -1]):
        print("Detected numeric target column.")
    else:
        print("Detected categorical target column.")
    
    # Separate features and target
    X = data.iloc[:, :-1]  # All columns except the last one
    y = data.iloc[:, -1]   # The last column
    
    # Implement Tree-based Methods for feature selection
    model = ExtraTreesClassifier()

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    model.fit(X, y)
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create a DataFrame to store feature importances
    feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    
    # Sort features by importance
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    
    # Select the top features based on a threshold or top N features
    threshold = 0.01  # Example threshold for importance
    selected_features = feature_importances[feature_importances['Importance'] > threshold]['Feature']
    
    # Check if no features are selected
    if selected_features.empty:
        print("No features met the threshold. Selecting top N features instead.")
        top_n = 5  # Example fallback to select top N features
        selected_features = feature_importances.head(top_n)['Feature']
    
    # Create a new DataFrame with only the selected features
    selected_data = data[selected_features.tolist() + [data.columns[-1]]]  # Add the target column back
    
    # Save the new DataFrame to a CSV file
    selected_data.to_csv(output_csv, index=False)
    print(f"Selected features data has been saved to {output_csv}")

if __name__ == "__main__":
    feature_selection()
