import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import rcParams
import warnings
warnings.filterwarnings("ignore")

# Load dataset
data = pd.read_csv("C:\Facultate\ISIA\Proiect\data.csv")
print(data.columns)

rcParams["figure.figsize"] = 10, 6
np.random.seed(42)

# Sterg liniile care au informatii lipsa
indexG = data[data['G'] == '?'].index
data.drop(indexG , inplace=True)

# Aleg coloana care defineste tipul tumorii
X = data.drop("K", axis=1)
y = data["K"]
# print(df.head(10))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, stratify=y, test_size=0.25, random_state=42
)
classifier = RandomForestClassifier(n_estimators=10)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(y_pred)