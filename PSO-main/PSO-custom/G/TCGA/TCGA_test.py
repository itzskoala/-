import pandas as pd

def print_csv_contents(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Print the columns
    print("Columns:")
    print(df.columns)
    
    # Print the rows
    print("\nRows:")
    print(df)
    
# Example usage
file_path = 'PSO-main/data/TCGA.csv'  # Replace with your actual file path
print_csv_contents(file_path)
