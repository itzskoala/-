import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

def select_relevant_features(input_csv, output_csv):
    # Load the dataset
    data = pd.read_csv(input_csv)

    # Separate the features and the target variable
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Apply SelectKBest with the Chi-Square test
    selector = SelectKBest(chi2, k='all')
    selector.fit(X, y)

    # Get the p-values for all features
    p_values = selector.pvalues_

    # Define a threshold for relevance (e.g., p-value < 0.05)
    threshold = 0.05

    # Get the indices of relevant features
    relevant_indices = [i for i, p in enumerate(p_values) if p < threshold]

    # Get the names of the relevant features
    relevant_features = X.columns[relevant_indices]

    # Save the relevant features to a CSV file
    with open(output_csv, 'w') as f:
        f.write(','.join(relevant_features))

if __name__ == "__main__":
    input_csv = input("Enter the path to the input CSV file: ")
    output_csv_input = input("Enter the path to the output CSV file: ")
    output_csv = 'PSO-main/PSO-custom/G/FeatureSelectionMethods/FilterMethods/Chi-Square Test/features/features_'+ output_csv_input + '.csv'
    select_relevant_features(input_csv, output_csv)
