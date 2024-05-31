# Adding more debugging information and making sure re-initialization happens correctly

import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

# Constants
NUM_PARTICLES = 30
NUM_ITERATIONS = 100
W = 0.5  # Inertia weight
C1 = 1.5  # Cognitive (particle) weight
C2 = 1.5  # Social (swarm) weight
SEARCH_SPACE_BOUNDS = (0, 1)  # Normalized bounds

# Read the dataset
df = pd.read_csv('data/Occupancy_Estimation.csv')

# Features and target
X = df[['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light', 
        'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound', 'S5_CO2', 'S5_CO2_Slope', 'S6_PIR', 'S7_PIR']]
y = df['Room_Occupancy_Count']

# Normalize the feature data to [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Perform Chi-Squared test
chi_scores, p_values = chi2(X_scaled, y)

# Create a DataFrame to display the scores and p-values
chi2_results = pd.DataFrame({'Feature': X.columns, 'Chi2 Score': chi_scores, 'P-value': p_values})
chi2_results = chi2_results.sort_values(by='Chi2 Score', ascending=False)

# Select top N features based on Chi-Squared scores
top_features = chi2_results['Feature'].values[:5]  # Adjust the number of top features as needed

# Filter the dataset to include only the top features
X_top = df[top_features].values

# Normalize the selected top features to [0, 1]
X_top_scaled = scaler.fit_transform(X_top)

# Initialize particles for each class
class_labels = df['Room_Occupancy_Count'].unique()
particles = {label: np.random.uniform(SEARCH_SPACE_BOUNDS[0], SEARCH_SPACE_BOUNDS[1], (NUM_PARTICLES, X_top_scaled.shape[1])) for label in class_labels}
velocities = {label: np.zeros((NUM_PARTICLES, X_top_scaled.shape[1])) for label in class_labels}

# Initialize best positions
pbest_positions = {label: particles[label].copy() for label in class_labels}
pbest_scores = {label: np.full(NUM_PARTICLES, np.inf) for label in class_labels}
gbest_positions = {label: np.zeros(X_top_scaled.shape[1]) for label in class_labels}
gbest_scores = {label: np.inf for label in class_labels}

# Fitness function
def fitness_function(particle, data):
    distances = np.linalg.norm(data - particle, axis=1)
    return np.mean(distances)

# Check the distribution of data for each class
for label in class_labels:
    print(f"Class {label} has {len(df[df['Room_Occupancy_Count'] == label])} data points.")

# Optimization loop
for iteration in range(NUM_ITERATIONS):
    for label in class_labels:
        data = X_top_scaled[df['Room_Occupancy_Count'] == label]
        if len(data) == 0:  # Skip if there's no data for the class
            print(f"No data for class {label}")
            continue
        for i in range(NUM_PARTICLES):
            fitness = fitness_function(particles[label][i], data)
            
            # Update personal best
            if fitness < pbest_scores[label][i]:
                pbest_scores[label][i] = fitness
                pbest_positions[label][i] = particles[label][i].copy()
            
            # Update global best
            if fitness < gbest_scores[label]:
                gbest_scores[label] = fitness
                gbest_positions[label] = particles[label][i].copy()
        
        # Update velocities and positions
        r1, r2 = np.random.rand(2)
        velocities[label] = (W * velocities[label] +
                             C1 * r1 * (pbest_positions[label] - particles[label]) +
                             C2 * r2 * (gbest_positions[label] - particles[label]))
        particles[label] += velocities[label]
        
        # Ensure particles stay within search space bounds
        particles[label] = np.clip(particles[label], SEARCH_SPACE_BOUNDS[0], SEARCH_SPACE_BOUNDS[1])

        # Re-initialize stuck particles
        if np.all(particles[label] == 0):
            particles[label] = np.random.uniform(SEARCH_SPACE_BOUNDS[0], SEARCH_SPACE_BOUNDS[1], (NUM_PARTICLES, X_top_scaled.shape[1]))

        # Print the particles for debugging
        if label == 0:
            print(f"Iteration {iteration}, Class {label} particles: {particles[label]}")
            print(f"Velocities: {velocities[label]}")
            print(f"Global Best Position: {gbest_positions[label]}, Global Best Score: {gbest_scores[label]}")
            print(f"Personal Best Scores: {pbest_scores[label]}")
            print(f"Current Particle Fitness: {[fitness_function(p, data) for p in particles[label]]}")

# Output the centroids
centroids = {label: gbest_positions[label] for label in class_labels}
print("Centroids for each class:")
for label, centroid in centroids.items():
    print(f"Class {label}: {centroid}")
