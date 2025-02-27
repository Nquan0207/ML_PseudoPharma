from train_set import train, test
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

X = train.drop(columns=["label"]) 
y = train["label"]  

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = train.drop(columns=['label'])
y_train = train['label']
X_test = test.drop(columns ='label')
y_test = test['label']

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

from deap import base, creator, tools, algorithms
import random

def fitness_function(individual):
    selected_features = [i for i in range(len(individual)) if individual[i] == 1]
    
    if len(selected_features) == 0: 
        return 0,

    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]
    
    model = GaussianNB()
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    
    return accuracy_score(y_test, y_pred),

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)  
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(X.columns))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", fitness_function)

pop = toolbox.population(n=20) 
algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=True)

best_ind = tools.selBest(pop, 1)[0]
selected_features = [X.columns[i] for i in range(len(best_ind)) if best_ind[i] == 1]
print("Selected features:", selected_features)

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

nb_model.fit(X_train_selected, y_train)
y_pred = nb_model.predict(X_test_selected)

accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Accuracy: {accuracy:.4f}")
