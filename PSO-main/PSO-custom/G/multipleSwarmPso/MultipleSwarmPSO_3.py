import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
NUM_PARTICLES = 30
NUM_ITERATIONS = 100
W = 0.5  # Inertia weight
C1 = 1.5  # Cognitive (particle) weight
C2 = 1.5  # Social (swarm) weight
SEARCH_SPACE_BOUNDS = (0, 100)  # Search space from 0 to 100

# Read the dataset
df = pd.read_csv('data/4Cluster2Ddataset.csv')

# Separate the data based on class labels
class_labels = df['Class'].unique()
class_data = {label: df[df['Class'] == label][['x', 'y']].values for label in class_labels}

# Initialize particles for each class
particles = {label: np.random.uniform(SEARCH_SPACE_BOUNDS[0], SEARCH_SPACE_BOUNDS[1], (NUM_PARTICLES, 2)) for label in class_labels}
velocities = {label: np.zeros((NUM_PARTICLES, 2)) for label in class_labels}

# Initialize best positions
pbest_positions = {label: particles[label].copy() for label in class_labels}
pbest_scores = {label: np.full(NUM_PARTICLES, np.inf) for label in class_labels}
gbest_positions = {label: np.zeros(2) for label in class_labels}
gbest_scores = {label: np.inf for label in class_labels}

# Fitness function
def fitness_function(particle, data):
    distances = np.linalg.norm(data - particle, axis=1)
    return np.mean(distances)

# Optimization loop
for iteration in range(NUM_ITERATIONS):
    for label in class_labels:
        data = class_data[label]
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

# Output the centroids
centroids = {label: gbest_positions[label] for label in class_labels}
print("Centroids for each class:")
for label, centroid in centroids.items():
    print(f"Class {label}: {centroid}")

# Visualize the centroids along with the data points
plt.figure(figsize=(10, 8))

# Plot each class data points and their centroid
for label in class_labels:
    data = class_data[label]
    plt.scatter(data[:, 0], data[:, 1], label=f'Class {label}')
    plt.scatter(centroids[label][0], centroids[label][1], marker='X', s=200, label=f'Centroid {label}', edgecolors='k')

plt.title('Cluster Data Points and Centroids')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
