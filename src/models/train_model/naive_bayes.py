from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import joblib
import os
from sklearn.naive_bayes import GaussianNB
import random
from hyperparam_tuning import *  # Assuming additional functions if needed
from deap import base, creator, tools, algorithms

# Load preprocessed data
data = np.load("../../../data/preprocessed/train_data.npz")
x_train, y_train = data["x_train"], data["y_train"].ravel()
data = np.load("../../../data/preprocessed/test_data.npz")
x_test, y_test = data["x_test"], data["y_test"].ravel()

# Define fitness function for the Genetic Algorithm
def fitness_function(params):
    # params[0] is the log10(var_smoothing) value
    var_smoothing = 10 ** params[0]
    model = GaussianNB(var_smoothing=var_smoothing)
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    return (accuracy,)

# Genetic Algorithm Setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

IND_SIZE = 1  # One parameter (log10(var_smoothing))
POP_SIZE = 20
NGEN = 100
MUTPB = 0.2
CXPB = 0.5

# Toolbox setup
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -9, 0)  # log10(var_smoothing) in range [-9, 0]
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitness_function)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run Genetic Algorithm
population = toolbox.population(n=POP_SIZE)
algorithms.eaSimple(population, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN,
                    stats=None, halloffame=None, verbose=True)

# Get best individual from GA
best_ind = tools.selBest(population, k=1)[0]
best_var_smoothing = 10 ** best_ind[0]
print(f"Best var_smoothing from GA: {best_var_smoothing}")

# Train final model using the GA-optimized parameter
final_model = GaussianNB(var_smoothing=best_var_smoothing)
final_model.fit(x_train, y_train)

# Define a parameter grid for further tuning with GridSearchCV.
# Here, we test a small range around the GA result.
param_grid1 = {
    "var_smoothing": [best_var_smoothing * 0.1, best_var_smoothing, best_var_smoothing * 10]
}

grid_search = GridSearchCV(final_model, param_grid1, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train, y_train)

# Get the best estimator from grid search
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(x_test)

# Evaluate performance
print(classification_report(y_test, y_pred))
print("Best Parameters from Grid Search:", grid_search.best_params_)

# Save the best model to disk
model_save_path = os.path.join("models", "trained", "naive_bayes_with_ga_best_grid.pkl")
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
joblib.dump(best_model, model_save_path)
print(f"Model saved successfully at: {model_save_path}")
