import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('data/Occupancy_Estimation.csv')

# Features and target
X = df[['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light', 
        'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound', 'S5_CO2', 'S5_CO2_Slope', 'S6_PIR', 'S7_PIR']]
y = df['Room_Occupancy_Count']

# Normalize the feature data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Perform Chi-Squared test
chi_scores, p_values = chi2(X_scaled, y)

# Create a DataFrame to display the scores and p-values
chi2_results = pd.DataFrame({'Feature': X.columns, 'Chi2 Score': chi_scores, 'P-value': p_values})
chi2_results = chi2_results.sort_values(by='Chi2 Score', ascending=False)

# Display the DataFrame directly
print(chi2_results)
