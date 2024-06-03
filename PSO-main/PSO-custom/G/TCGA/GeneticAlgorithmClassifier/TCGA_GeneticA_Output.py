import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random

# Load the dataset
file_path = 'C:/Users/guill/Downloads/TCGA.csv'
tcga_data = pd.read_csv(file_path)

# Drop any unnamed columns that are likely unnecessary
tcga_data = tcga_data.loc[:, ~tcga_data.columns.str.contains('^Unnamed')]

# Separate features and target
X = tcga_data.drop('Class', axis=1)
y = tcga_data['Class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parameters for the genetic algorithm
population_size = 50
generations = 10
mutation_probability = 0.2
crossover_probability = 0.5

def create_individual(n_features):
    return [random.randint(0, 1) for _ in range(n_features)]

def create_population(pop_size, n_features):
    return [create_individual(n_features) for _ in range(pop_size)]

def evaluate_individual(individual):
    selected_features = [index for index in range(len(individual)) if individual[index] == 1]
    if len(selected_features) == 0:
        return 0,
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train.iloc[:, selected_features], y_train)
    predictions = clf.predict(X_test.iloc[:, selected_features])
    return accuracy_score(y_test, predictions),

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

def mutate(individual):
    for i in range(len(individual)):
        if random.random() < mutation_probability:
            individual[i] = 1 - individual[i]

def genetic_algorithm():
    population = create_population(population_size, X_train.shape[1])
    for generation in range(generations):
        population = sorted(population, key=lambda ind: evaluate_individual(ind), reverse=True)
        next_generation = population[:int(0.2 * population_size)]
        while len(next_generation) < population_size:
            if random.random() < crossover_probability:
                parent1, parent2 = random.sample(population[:int(0.5 * population_size)], 2)
                offspring1, offspring2 = crossover(parent1, parent2)
                next_generation += [offspring1, offspring2]
            else:
                individual = random.choice(population[:int(0.5 * population_size)])
                mutate(individual)
                next_generation.append(individual)
        population = next_generation
    best_individual = max(population, key=lambda ind: evaluate_individual(ind))
    return best_individual

best_individual = genetic_algorithm()
selected_features = [index for index in range(len(best_individual)) if best_individual[index] == 1]

# Get the selected feature names
selected_gene_names = X_train.columns[selected_features]

# Output the selected features to a file
output_file_path = 'PSO-main/PSO-custom/G/TCGA/RandomForestClassifier/results/results_TCGA_GA.csv'
with open(output_file_path, 'w') as f:
    for feature in selected_gene_names:
        f.write(f"{feature}\n")

print(f"Selected features have been saved to {output_file_path}")
