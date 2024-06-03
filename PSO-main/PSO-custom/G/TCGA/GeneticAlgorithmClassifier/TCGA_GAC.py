"""
Author: [Guillermo Munoz]
Date: [06/03/2024]
Description: This script performs Genetic Algorithm-Based Feature Selection on the TCGA dataset.

- The script loads the TCGA dataset, removes unnecessary columns, and separates features and target.
- It utilizes the DEAP library to implement the genetic algorithm for feature selection.
- Logistic Regression is used as the classifier to evaluate the performance of selected features.
- The selected feature names are saved to a specified CSV file.

Steps:
1. Initialization of the population with random feature subsets.
2. Fitness evaluation using Logistic Regression to measure the accuracy of each subset.
3. Selection of the best individuals to form a mating pool.
4. Crossover and mutation to create new offspring.
5. Replacement of the population with new offspring.
6. Extraction of the best subset of features and output to a CSV file.

Required Libraries:
- pandas
- numpy
- deap
- scikit-learn

Usage:
1. Ensure the necessary libraries are installed.
2. Set the correct file paths for input dataset and output CSV.
3. Run the script to perform feature selection and save the results.

"""

import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
file_input = input("Please enter file input path: ")
file_path = file_input
tcga_data = pd.read_csv(file_path)

# Drop any unnamed columns that are likely unnecessary
tcga_data = tcga_data.loc[:, ~tcga_data.columns.str.contains('^Unnamed')]

# Split the data into features and target
class_input = input("Please enter target class name: ")
X = tcga_data.drop(columns=[class_input])
y = tcga_data[class_input]

# Define evaluation function
def evaluate(individual):
    selected_features = [index for index, val in enumerate(individual) if val == 1]
    if len(selected_features) == 0:
        return 0,
    
    X_selected = X.iloc[:, selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred),

# Define the genetic algorithm
toolbox = base.Toolbox()
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Attribute generator: define 'n' features as a binary string
n_features = X.shape[1]
toolbox.register("attr_bool", np.random.randint, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Genetic Algorithm parameters
population = toolbox.population(n=50)
ngen = 40
cxpb = 0.5
mutpb = 0.2

# Run the genetic algorithm
algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=False)

# Extracting the best individual
best_individual = tools.selBest(population, 1)[0]
selected_features = [index for index, val in enumerate(best_individual) if val == 1]

# Output the selected feature names to a file
output_path_input = input("Please enter output csv file name: ")
output_file_path = 'PSO-main/PSO-custom/G/TCGA/GeneticAlgorithmClassifier/results/'+ output_path_input + '.csv'
with open(output_file_path, 'w') as f:
    f.write(','.join(X.columns[selected_features]))

print(f"Selected feature names have been saved to {output_file_path}")
