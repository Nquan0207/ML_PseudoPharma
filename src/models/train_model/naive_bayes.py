from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
import joblib
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import random
from hyperparam_tuning import *
from deap import base, creator, tools, algorithms


# Load preprocessed data
data = np.load("../../../data/preprocessed/train_data.npz")
x_train, y_train = data["x_train"], data["y_train"].ravel()
data = np.load("../../../data/preprocessed/val_data.npz")
x_val, y_val = data["x_val"], data["y_val"].ravel()
data = np.load("../../../data/preprocessed/test_data.npz")
x_test, y_test = data["x_test"], data["y_test"].ravel()

# Define fitness function
def fitness_function(params):
    var_smoothing = 10 ** params[0]
    model = GaussianNB(var_smoothing=var_smoothing)
    model.fit(x_train, y_train)
    accuracy = model.score(x_val, y_val)
    return (accuracy,)

# Genetic Algorithm Setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

IND_SIZE = 1  # One parameter to optimize
POP_SIZE = 20
NGEN = 100
MUTPB = 0.2
CXPB = 0.5

# Toolbox setup
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -9, 0)  # log10(var_smoothing)
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

# Get best individual
best_ind = tools.selBest(population, k=1)[0]
best_var_smoothing = 10 ** best_ind[0]
print(f"Best var_smoothing: {best_var_smoothing}")

# Train final model
final_model = GaussianNB(var_smoothing=best_var_smoothing)
# final_model.fit(x_train, y_train)
# test_accuracy = final_model.score(x_test, y_test)
# print(f"Test Accuracy: {test_accuracy}")


grid_search = GridSearchCV(final_model, param_grid1, cv=3, scoring='accuracy', n_jobs=-1)

# Fit the model (this finds the best hyperparameters)
grid_search.fit(x_train_with_stats, y_train)

# Get best model
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(x_test_with_stats)

# Evaluate performance
print(classification_report(y_test, y_pred))
print("Best Parameters:", grid_search.best_params_)
# save model
model_save_path = os.path.join("models", "trained", "naive_bayes_with_ga_best_grid.pkl")

# Ensure the directory exists
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Save the model
joblib.dump(best_model, model_save_path)
print(f"Model saved successfully at: {model_save_path}")