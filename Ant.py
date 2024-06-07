import pandas as pd
import numpy as np
import random
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


class AntColonyFeatureSelector:
    def __init__(self, file_path, num_ants=10, num_iterations=100, evaporation_rate=0.5, alpha=1, beta=2):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.pheromone = np.ones((self.data.shape[1] - 1, self.data.shape[1] - 1))
        self.target = self.data.columns[-1]  # Assuming the target variable is the last column

        # Min-Max scaling of the features
        self.scale_features()

    def scale_features(self):
        scaler = MinMaxScaler()
        feature_columns = self.data.columns[:-1]  # All columns except the last one
        self.data[feature_columns] = scaler.fit_transform(self.data[feature_columns])

    def evaluate_path(self, path):
        """Evaluate the quality of the selected path."""
        selected_features = [index for index, value in enumerate(path) if value == 1]
        
        if not selected_features:  # Avoid empty feature set
            return -np.inf
        
        X = self.data.iloc[:, selected_features]
        y = self.data[self.target]
        
        # Use LogisticRegression and cross-validation to evaluate feature set
        classifier = LogisticRegression(max_iter=1000)
        scores = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
        return scores.mean()

    def select_features(self):
        best_path = None
        best_score = -np.inf

        for iteration in range(self.num_iterations):
            paths = []
            scores = []

            for ant in range(self.num_ants):
                path = self.generate_path()
                score = self.evaluate_path(path)
                paths.append(path)
                scores.append(score)

                if score > best_score:
                    best_score = score
                    best_path = path

            self.update_pheromone(paths, scores)

        selected_features = [i for i in range(len(best_path)) if best_path[i] == 1]
        return selected_features

    def generate_path(self):
        path = np.zeros(self.data.shape[1] - 1)
        for i in range(len(path)):
            if random.random() < 0.5:
                path[i] = 1
        return path

    def update_pheromone(self, paths, scores):
        self.pheromone *= (1 - self.evaporation_rate)

        for path, score in zip(paths, scores):
            for i in range(len(path)):
                if path[i] == 1:
                    self.pheromone[i] += score

    def run(self):
        selected_features = self.select_features()
        modified_data = self.data.iloc[:, selected_features]
        # Include the target class in the output file
        modified_data[self.target] = self.data[self.target]
        modified_data.to_csv('Ant_FS.csv', index=False)
        return modified_data

# Usage example
if __name__ == "__main__":
    file_path = input("Please enter the path to your dataset: ")
    ant_colony_fs = AntColonyFeatureSelector(file_path)
    selected_data = ant_colony_fs.run()
    print("Selected features dataset saved to 'Ant_FS.csv'")
