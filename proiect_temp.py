import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.preprocessing import LabelEncoder

# importul fisierului cu date

df = pd.read_csv('C:\Facultate\ISIA\Proiect\data.csv') 
df.head(7)

# afisare date si dimensiune

print(df)
print(df.shape)

# algoritm pentru afisarea liniilor unde apare '?'

searchfile = open('C:\Facultate\ISIA\Proiect\data.csv', "rt")
reader = csv.reader(searchfile, delimiter = ',')
for row in reader:
    for field in row:
            if field == '?':
                print(row)

# numar total cazuri 2 for benign, 4 for malignant + tipurile de date

print(df['2.1'].value_counts())

# 0 - Benign si 1 - Malign

#labelencoder_Y = LabelEncoder()
#df.iloc[:,10] = labelencoder_Y.fit_transform(df.iloc[:,10].values)

# 

sns.pairplot(df.iloc[:,1:10])
sns.heatmap(df.iloc[:,1:12].corr())
