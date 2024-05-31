import pandas as pd

# Load the dataset
df = pd.read_csv('data/Occupancy_Estimation.csv')

# Print the number of data points for each class
class_counts = df['Room_Occupancy_Count'].value_counts()
print(class_counts)
