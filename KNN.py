import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

le = preprocessing.LabelEncoder()

data = pd.read_csv('car.data', sep=",", header=None)

data.columns = ["buying", "maint", "doors", "persons","lug_boot","safety","class"]

#print(data)

data['buying']=le.fit_transform(data['buying'])
data['maint']=le.fit_transform(data['maint'])
data['doors']=le.fit_transform(data['doors'])
data['persons']=le.fit_transform(data['persons'])
data['lug_boot']=le.fit_transform(data['lug_boot'])
data['safety']=le.fit_transform(data['safety'])
data['class']=le.fit_transform(data['class'])

#print(data)

Class=data['class']

del data['class']

X_train, X_test, y_train, y_test = train_test_split(data, Class, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=8)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("KNN")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred,average='macro'))
print("Precision:",metrics.precision_score(y_test, y_pred,average='macro'))
print("\n")