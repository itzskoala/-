from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import f_classif, SelectKBest

def load_and_prepare_data(df):
    # Separate features and target
    X = df.iloc[:, :-1]  # All columns except the last one
    y = df.iloc[:, -1]   # The last column
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Convert categorical data to numerical values
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column])
    
    # Scale the features
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X, y

def perform_anova_feature_selection(X, y, k=10):
    # Perform ANOVA feature selection
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    
    # Get the columns selected
    selected_columns = X.columns[selector.get_support()]
    return selected_columns

def save_selected_features(df, selected_features, output_path):
    # Filter the dataframe to keep only the selected features and the target column
    selected_data = df[selected_features.tolist() + [df.columns[-1]]]
    
    # Save the selected features to a new CSV file
    selected_data.to_csv(output_path, index=False)
    
    print(f"Selected features have been saved to {output_path}")

def main():
    # Ask the user for the CSV file path
    input_file_path = input("Enter the path to the CSV file: ")
    
    # Load the dataset
    df = pd.read_csv(input_file_path)
    
    # Load and prepare the data
    X, y = load_and_prepare_data(df)
    
    # Perform ANOVA feature selection
    selected_features = perform_anova_feature_selection(X, y)
    
    # Define the output file path
    output_file_path = 'ANOVA_FS.csv'
    
    # Save selected features to a new CSV file
    save_selected_features(df, selected_features, output_file_path)

if __name__ == "__main__":
    main()
