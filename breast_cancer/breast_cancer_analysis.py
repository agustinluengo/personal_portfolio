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

#Reviewing correlation between variables
correlation = psd_data.corr()

sns.heatmap(data=correlation)
plt.show()

#extracting the variables with the highest correlation
corr_data = pd.DataFrame(correlation)
corr_data[corr_data['diagnosis']*100 > 70]

high_corr_psd_data = psd_data[['diagnosis','radius_mean','perimeter_mean','area_mean','concave points_mean','radius_worst','perimeter_worst', 'area_worst', 'concave points_worst']]

#Plotting general view of the data
sns.pairplot(data=high_corr_psd_data,hue='diagnosis')
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier, plot_
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report

# Split the dataset into features (X) and target variable (y)
X = high_corr_psd_data.drop(columns=['diagnosis'])  # Features
y = high_corr_psd_data['diagnosis']  # Target variable

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_accuracies = {}
model_precision = {}
model_recall = {}
model_f1 = {}

dt_model = DecisionTreeClassifier()
#dt_model.get_params()
#criterion: Gini/Entropy (default: Gini)
#splitter: best/random (default: Best)
#max_depth: N (default:None)
#random_state: N (default:None)
dt_model.fit(X_train, y_train)
#features = list(feature_importance[0]>0].index)
feature_importances = pd.DataFrame(dt_model.feature_importances_,index=X.columns).sort_values(by=0,ascending=False)
feature_importances.head(10).plot(kind='bar')
plt.show()

# plot_tree(dt_model,
#            feature_names=X.columns,
#            class_names={0:'M',1:'B'},
#            filled=True,
#            fontsize=12
#            )
#plt.show()

text_representation = export_text(dt_model)
print(text_representation)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

svc_model = SVC(kernel='linear',decision_function_shape='ovo')
#kernel options: linear, rbf, polynomial, sigmoid tanh
#decision_function_shape='ovo' means one vs one classifiers / one vs rest classifiers
svc_model.fit(X_train, y_train)


nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

knn_model = KNeighborsClassifier(n_neighbors=2)
knn_model.fit(X_train, y_train)

models = {'dt_model':dt_model,
 'rf_model':rf_model,
'svc_model':svc_model,
'nb_model':nb_model,
'knn_model':knn_model}

for model_name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    model_accuracies[model_name] = accuracy*100
    model_precision[model_name] = precision*100
    model_recall[model_name] = recall*100
    model_f1[model_name] = f1*100

acc = pd.DataFrame.from_dict(model_accuracies, orient='Index', columns=['Accuracy'])
prec = pd.DataFrame.from_dict(model_precision, orient='Index', columns=['Precision'])
rec = pd.DataFrame.from_dict(model_recall, orient='Index', columns=['Recall'])
f1 = pd.DataFrame.from_dict(model_f1, orient='Index', columns=['F1_Score'])
report = pd.concat([acc, prec, rec, f1], axis=1, join='inner')


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Create a PCA instance
pca = PCA(n_components=2)
# Fit PCA to the scaled data
pca.fit(X_scaled)
# Transform the data to the new feature space
X_pca = pca.transform(X_scaled)


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

svc_model.fit(X_train, y_train)
svc_pred = svc_model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test,y_pred)
# f1 = f1_score(y_test,y_pred)

# Plot decision boundary in the reduced feature space
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

Z = svc_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, marker='o', edgecolors='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('SVC Decision Boundary after PCA')
plt.show()

# Print explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)
# Print cumulative explained variance ratio
print("Cumulative explained variance ratio:", pca.explained_variance_ratio_.cumsum())


confusion_matrix(y_test, svc_pred, labels=[0,1])
