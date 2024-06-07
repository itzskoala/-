import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

def load_and_prepare_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Separate features and target
    X = data.iloc[:, :-1]  # All columns except the last one
    y = data.iloc[:, -1]   # The last column
    
    # Scale the features
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X, y

def evaluate_gbm(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize the GBM model
    gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    
    # Train the model
    gbm.fit(X_train, y_train)
    
    # Evaluate the model using cross-validation
    scores = cross_val_score(gbm, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation Accuracy: {scores.mean():.4f}")
    
    return gbm

def plot_feature_importance(model, X):
    # Get feature importance
    importance = model.feature_importances_
    
    # Create a DataFrame for better visualization
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    
    # Plot the feature importance
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance - Gradient Boosting Machine')
    plt.gca().invert_yaxis()
    plt.show()
    
    return feature_importance

def save_selected_features(df, selected_features, output_path):
    # Filter the dataframe to keep only the selected features and the target column
    selected_data = df[selected_features.tolist() + [df.columns[-1]]]
    
    # Save the selected features to a new CSV file
    selected_data.to_csv(output_path, index=False)
    
    print(f"Selected features have been saved to {output_path}")

def main():
    # Ask the user for the CSV file path
    file_path = input("Please enter the path to your dataset: ")
    
    # Load and prepare the data
    X, y = load_and_prepare_data(file_path)
    
    # Evaluate the model using GBM
    gbm_model = evaluate_gbm(X, y)
    
    # Plot and get feature importance
    feature_importance = plot_feature_importance(gbm_model, X)
    
    # Define the output file path
    output_file_path = 'GBM_FS.csv'
    
    # Save selected features to a new CSV file
    save_selected_features(pd.read_csv(file_path), feature_importance['Feature'], output_file_path)

if __name__ == "__main__":
    main()
