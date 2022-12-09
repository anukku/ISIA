# PROIECT ISIA
# CIOBANU MATEI-CIPRIAN 421E

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

procent_in_bag = np.array ([0.25 , 0.5 , 0.85]) # definesc valorile pe care vreau sa le alternez in array-uri
nr_dimensiuni = np.array ([0.1 , 0.5 , 0.8]) # fac asta si pentru linii (procent_in_bag) si pentru coloane (nr_dimensiuni) 

for i in range (len(procent_in_bag)) : # alternez valorile
    for j in range (len(nr_dimensiuni)) : # alternez valorile
        data = pd.read_csv("C:\Facultate\ISIA\Proiect\data.csv", header = None, # incarc baza de date, precizand calea fisierului
                    names = ['ID','B','C','D','E','F','G','H','I','J','class']) # atribui coloanelor denumiri
        np.random.seed(10) # functia seteaza 'state of randomness'
        indexG = data[data['G'] == '?'].index # parcurg celulele bazei de date pentru a descoperi celulele unde exista 
        data.drop(indexG , inplace=True) # sterg din baza de date celulele in care am gasit '?', iar parametrul 'inplace' este setat 'true' pentru a returna DataFrame-ul fara aceste celule
        X = data.drop("class", axis=1) # dau slice coloanei care imi indica daca tumoarea este maligna sau benigna
        y = data["class"] # salvez coloana in alta variabila
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.25, random_state=None) # separ datele de testare de cele de antrenare
        classifier = RandomForestClassifier(n_estimators = 10, max_samples = procent_in_bag[i], max_features = nr_dimensiuni[j])
        classifier.fit(X_train, y_train) # antrenarea
        y_pred = classifier.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred)) # acuratetea din testare + cat la suta din prezicere e corecta
        print(y_pred)

# BIBLIOGRAPHY
# https://stats.stackexchange.com/questions/158583/what-does-node-size-refer-to-in-the-random-forest
# https://stackoverflow.com/questions/58325781/random-forest-in-bag-and-node-dimensions
# https://www.freecodecamp.org/news/how-to-use-the-tree-based-algorithm-for-machine-learning/