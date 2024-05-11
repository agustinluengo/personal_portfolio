# Importing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


#Loading your dataset
data = pd.read_csv("C:/Users/luengoag/Desktop/Private/Python/breast_cancer/breast_cancer.csv")

#Reviewing the dataset
data.shape
data.columns
data.dtypes
data.isnull().sum()

#Removing extra column with Null values
data = data.drop(columns='Unnamed: 32',axis='columns')

#Encoding the diagnosis column to be used as a numeric variable.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

psd_data = data.copy().drop(columns='diagnosis', axis="columns")
psd_data['diagnosis'] = le.fit_transform(data['diagnosis'])

#list(le.classes_)

#Checking distribution of Malicious and Benign samples
sns.countplot(data=data, x='diagnosis', hue='diagnosis')
plt.show()

#Plotting general view of the data
sns.pairplot(data=data,hue='diagnosis')
plt.show()

#Reviewing correlation between variables
correlation = psd_data.corr()

sns.heatmap(data=correlation)
plt.show()

#extracting the variables with the highest correlation
corr_data = pd.DataFrame(correlation)
corr_data[corr_data['diagnosis']*100 > 70]

high_corr_psd_data = psd_data[['diagnosis','radius_mean','perimeter_mean','area_mean','concave points_mean','radius_worst','perimeter_worst', 'area_worst', 'concave points_worst']]

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Split the dataset into features (X) and target variable (y)
X = data.drop(columns=['target_column'])  # Features
y = data['target_column']  # Target variable

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a list of ML models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Linear Regression": LinearRegression(),
    "Logistic Regression": LogisticRegression(),
    "SVC": SVC(),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier()
}

# Dictionary to store model accuracies
model_accuracies = {}

# Loop through each model
for name, model in models.items():
    # Fit the model on the training data
    model.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store the accuracy in the dictionary
    model_accuracies[name] = accuracy

# Print model accuracies
for name, accuracy in model_accuracies.items():
    print(f"{name}: Accuracy = {accuracy}")

# Choose the model with the highest accuracy
best_model = max(model_accuracies, key=model_accuracies.get)
print(f"\nThe best model is: {best_model} with accuracy = {model_accuracies[best_model]}")
