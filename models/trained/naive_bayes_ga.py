from train_set import train, test
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from deap import base, creator, tools, algorithms
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import random

X = train.drop(columns=["label"]) 
y = train["label"]  

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = train.drop(columns=['label'])
y_train = train['label']
X_test = test.drop(columns ='label')
y_test = test['label']

# nb_model = GaussianNB()
# nb_model.fit(X_train, y_train)

# y_pred = nb_model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.4f}")

### With GA

X_train_global = X_train.iloc[:, :2]  # Mean, variance của toàn ảnh (luôn giữ lại)
X_train_blocks = X_train.iloc[:, 2:]  # Mean, variance của từng block (GA sẽ tối ưu)

X_test_global = X_test.iloc[:, :2]
X_test_blocks = X_test.iloc[:, 2:]

def fitness_function(individual):

    selected_features = [i for i in range(len(individual)) if individual[i] == 1]
    
    if len(selected_features) == 0:  
        return 0,

    X_train_selected = X_train_blocks.iloc[:, selected_features]
    X_test_selected = X_test_blocks.iloc[:, selected_features]

    X_train_final = pd.concat([X_train_global.reset_index(drop=True), X_train_selected.reset_index(drop=True)], axis=1)
    X_test_final = pd.concat([X_test_global.reset_index(drop=True), X_test_selected.reset_index(drop=True)], axis=1)

    model = GaussianNB()
    model.fit(X_train_final, y_train)
    y_pred = model.predict(X_test_final)
    
    return accuracy_score(y_test, y_pred),


creator.create("FitnessMax", base.Fitness, weights=(1.0,)) 
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
random.seed(1000)
toolbox.register("attr_bool", random.randint, 0, 1)  # Choose integer attribute between 0 and 1
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X_train_blocks.shape[1])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.65)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", fitness_function)

pop = toolbox.population(n=20)

algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=True)

best_ind = tools.selBest(pop, 1)[0]
selected_features = [X_train_blocks.columns[i] for i in range(len(best_ind)) if best_ind[i] == 1]
print("Selected block features:", selected_features, len(selected_features))

X_train_selected = X_train_blocks[selected_features]
X_test_selected = X_test_blocks[selected_features]

X_train_final = pd.concat([X_train_global.reset_index(drop=True), X_train_selected.reset_index(drop=True)], axis=1)
X_test_final = pd.concat([X_test_global.reset_index(drop=True), X_test_selected.reset_index(drop=True)], axis=1)

nb_model = GaussianNB()
nb_model.fit(X_train_final, y_train)
y_pred = nb_model.predict(X_test_final)

accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Accuracy: {accuracy:.4f}")