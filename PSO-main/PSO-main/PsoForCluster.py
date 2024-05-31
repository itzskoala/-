import numpy as np
import pandas as pd
import random
import time

# Measure time taken for optimization
start_time = time.time()

# Load the dataset from CSV file into a pandas DataFrame
#dataset = pd.read_csv('data/4Cluster2Ddataset.csv')

dataset = pd.read_csv('data/Occupancy_Estimation.csv')

# Define PSO parameters
num_particles = 20
max_iterations = 100
inertia_weight = 0.9
c1 = 2  # cognitive coefficient
c2 = 2  # social coefficient
search_space = [(0, 100), (0, 100)]  # Search space boundaries
convergence_threshold = 1e-6  # Termination threshold for PSO convergence

# Define PSO function
def PSO(data, label):
    positions = np.array([[random.uniform(search_space[0][0], search_space[0][1]), random.uniform(search_space[1][0], search_space[1][1])] for _ in range(num_particles)])
    velocities = np.zeros_like(positions)
    pbest_positions = positions.copy()
    pbest_scores = np.array([fitness(data, pos, label) for pos in positions])
    gbest_position = positions[pbest_scores.argmin()]
    gbest_score = pbest_scores.min()
    
    # PSO main loop
    for iteration in range(max_iterations):
        # Calculate improvement in global best position
        prev_gbest_position = gbest_position.copy()
        
        for i in range(num_particles):
            r1, r2 = random.random(), random.random()
            velocities[i] = (inertia_weight * velocities[i] +
                             c1 * r1 * (pbest_positions[i] - positions[i]) +
                             c2 * r2 * (gbest_position - positions[i]))
            positions[i] += velocities[i]
            
            # Update personal best
            score = fitness(data, positions[i], label)
            if score < pbest_scores[i]:
                pbest_positions[i] = positions[i].copy()
                pbest_scores[i] = score
                
        # Update global best
        best_particle_idx = pbest_scores.argmin()
        if pbest_scores[best_particle_idx] < gbest_score:
            gbest_position = pbest_positions[best_particle_idx].copy()
            gbest_score = pbest_scores[best_particle_idx]
        
        # Check convergence
        if np.linalg.norm(gbest_position - prev_gbest_position) < convergence_threshold:
            break
    
    print(f"Centroid found for label {label}:", gbest_position)
    return gbest_position

# Define fitness function
def fitness(data, position, label):
    centroid_x, centroid_y = position
    distances = np.sqrt(((data[data['Class'] == label][['x', 'y']] - position)**2).sum(axis=1)).sum()
    return distances

# Perform PSO for each label
centroid_A = PSO(dataset, 'A')
centroid_B = PSO(dataset, 'B')
centroid_C = PSO(dataset, 'C')
centroid_D = PSO(dataset, 'D')

# Output
print("Centroid for label A:", centroid_A)
print("Centroid for label B:", centroid_B)
print("Centroid for label C:", centroid_C)
print("Centroid for label D:", centroid_D)

# Print time taken
print("Time taken:", time.time() - start_time, "seconds")

