import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

le = preprocessing.LabelEncoder()

data = pd.read_csv('car.data', sep=",", header=None)

data.columns = ["buying", "maint", "doors", "persons","lug_boot","safety","class"]


data['buying']=le.fit_transform(data['buying'])
data['maint']=le.fit_transform(data['maint'])
data['doors']=le.fit_transform(data['doors'])
data['persons']=le.fit_transform(data['persons'])
data['lug_boot']=le.fit_transform(data['lug_boot'])
data['safety']=le.fit_transform(data['safety'])
data['class']=le.fit_transform(data['class'])

Class=data['class']

del data['class']

X_train, X_test, y_train, y_test = train_test_split(data, Class, test_size=0.3, random_state=42)

RF = RandomForestClassifier(max_depth=10, n_estimators=100, max_features=5)

RF.fit(X_train,y_train)

y_pred=RF.predict(X_test)
print("Random Forest")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred,average='macro'))
print("Precision:",metrics.precision_score(y_test, y_pred,average='macro'))
print("\n")