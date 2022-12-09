import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

procent_in_bag = 0.85 # definesc valorile pe care vreau sa le alternez in array-uri
nr_dimensiuni = 80 # fac asta si pentru test_size si pentru n_estimators

data = pd.read_csv("C:\Facultate\ISIA\Proiect\data.csv", header = None, # incarc baza de date, precizand calea fisierului
    names = ['ID','B','C','D','E','F','G','H','I','J','class']) # atribui coloanelor denumiri
indexG = data[data['G'] == '?'].index # parcurg celulele bazei de date pentru a descoperi celulele unde exista 
data.drop(indexG , inplace=True) # sterg din baza de date celulele in care am gasit '?', iar parametrul 'inplace' este setat 'true' pentru a returna DataFrame-ul fara aceste celule
X = data.drop("class", axis=1) # dau slice coloanei care imi indica daca tumoarea este maligna sau benigna
y = data["class"] # salvel coloana in alta variabila
scaler = StandardScaler() # Standardize features by removing the mean and scaling to unit variance.
X_scaled = scaler.fit_transform(X) 
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, stratify=y, test_size = 0.25, random_state=None
    )
classifier = RandomForestClassifier(n_estimators = 10, max_samples = procent_in_bag, max_features = nr_dimensiuni)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(y_pred)