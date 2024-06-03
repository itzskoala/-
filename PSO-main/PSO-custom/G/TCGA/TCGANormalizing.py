
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the CSV file
file_path = 'C:/Users/guill/Downloads/TCGA.csv'
data = pd.read_csv(file_path)

# Drop unnecessary columns
data_cleaned = data.drop(columns=['Unnamed: 0', 'Unnamed: 20532', 'Class'])

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the data
data_normalized = scaler.fit_transform(data_cleaned)

# Convert the normalized data back to a DataFrame
data_normalized_df = pd.DataFrame(data_normalized, columns=data_cleaned.columns)

# Add back the 'Unnamed: 0', 'Unnamed: 20532', and 'Class' columns
data_normalized_df.insert(0, 'Unnamed: 0', data['Unnamed: 0'])
data_normalized_df['Unnamed: 20532'] = data['Unnamed: 20532']
data_normalized_df['Class'] = data['Class']

# Save the normalized data to a new CSV file
output_file_path = 'TCGA_normalized.csv'
data_normalized_df.to_csv(output_file_path, index=False)
