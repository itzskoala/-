import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    # Drop columns with all NaN values
    data.dropna(axis=1, how='all', inplace=True)
    return data

def prepare_data(data):
    # Separate features and target
    X = data.iloc[:, :-1]  # all columns except the last one
    y = data.iloc[:, -1]   # the last column
    return X, y

def train_model(X, y):
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Fit the model
    clf.fit(X_train, y_train)
    
    # Predict the test set
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Perform cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5)
    mean_cv_accuracy = cv_scores.mean()
    
    return clf, accuracy, mean_cv_accuracy

def save_results(model_name, accuracy, mean_cv_accuracy, output_path):
    # Save results to CSV
    results = pd.DataFrame([{
        "Model": model_name,
        "Accuracy": accuracy,
        "Mean Cross-Validated Accuracy": mean_cv_accuracy
    }])
    results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    # Get input and output file paths from user
    input_file = input("Enter the path to the input CSV file: ")
    output_file = input("Enter the path to save the results CSV file: ")
    
    # Load and clean data
    data = load_data(input_file)
    data = clean_data(data)
    
    # Prepare data
    X, y = prepare_data(data)
    
    # Train model and get accuracy
    model, accuracy, mean_cv_accuracy = train_model(X, y)
    
    # Save results
    save_results("Random Forest", accuracy, mean_cv_accuracy, output_file)
