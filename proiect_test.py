import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# figure size in inches
rcParams["figure.figsize"] = 10, 6
np.random.seed(42)

# ignore the lines with missing information
indexG = data[data['G'] == '?'].index
data.drop(indexG , inplace=True)


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
# feature_importances_df = pd.DataFrame(
# {"feature": list(X.columns), "importance": classifier.feature_importances_}
# ).sort_values("importance", ascending=False)
# feature_importances_df
# sns.barplot(x=feature_importances_df.feature, y=feature_importances_df.importance)
# plt.xlabel("Feature Importance Score")
# plt.ylabel("Features")
# plt.title("Visualizing Important Features")
# plt.xticks(
# rotation=45, horizontalalignment="right", fontweight="light", fontsize="x-large"
# )
# plt.show()
# # load data with selected features

# X = data.drop("K", axis=1)
# y = data["K"]

# # standardize the dataset
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # split into train and test set
# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y, stratify=y, test_size=0.25, random_state=42
# )

# # Create a Random Classifier
# clf = RandomForestClassifier(n_estimators=10)

# # Train the model using the training sets
# clf.fit(X_train, y_train)

# # prediction on test set
# y_pred = clf.predict(X_test)

# # Calculate Model Accuracy,
# print("Accuracy:", accuracy_score(y_test, y_pred))