import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

procent_in_bag = np.array ([0.25 , 0.5 , 0.85]) # definesc valorile pe care vreau sa le alternez in array-uri
nr_dimensiuni = np.array ([10 , 50 , 80]) # fac asta si pentru test_size si pentru n_estimators

for i in range (len(procent_in_bag)) : # alternez valorile pentru test_size
    for j in range (len(nr_dimensiuni)) : # alternez valorile pentru n_estimators
        data = pd.read_csv("C:\Facultate\ISIA\Proiect\data.csv", header = None, # incarc baza de date, precizand calea fisierului
                    names = ['ID','B','C','D','E','F','G','H','I','J','class']) # atribui coloanelor denumiri
        np.random.seed(42) # functia seteaza 'state of randomness'
        indexG = data[data['G'] == '?'].index # parcurg celulele bazei de date pentru a descoperi celulele unde exista 
        data.drop(indexG , inplace=True) # sterg din baza de date celulele in care am gasit '?', iar parametrul 'inplace' este setat 'true' pentru a returna DataFrame-ul fara aceste celule
        X = data.drop("class", axis=1) # dau slice coloanei care imi indica daca tumoarea este maligna sau benigna
        y = data["class"] # salvel coloana in alta variabila
        scaler = StandardScaler() # Standardize features by removing the mean and scaling to unit variance.
        X_scaled = scaler.fit_transform(X) 
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, stratify=y, test_size=procent_in_bag[i], random_state=42
        )
        classifier = RandomForestClassifier(n_estimators = nr_dimensiuni[j])
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(y_pred)