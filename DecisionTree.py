from sklearn import tree, preprocessing
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv('car.data', sep=",")

data.columns = ["buying", "maint", "doors", "persons","lug_boot","safety","class"]

le = preprocessing.LabelEncoder()

X= data.apply(le.fit_transform)


data1 = X.drop(columns="class")

target=X['class']

X_train, X_test, y_train, y_test = train_test_split(data1, target, test_size=0.43, random_state=42)

clf = tree.DecisionTreeClassifier(random_state=42)

clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Decision Tree")
print('Accuracy: {0:.4f}'.format(accuracy_score(y_test, y_pred)))
print("Recall:",metrics.recall_score(y_test, y_pred,average='macro'))
print("Precision:",metrics.precision_score(y_test, y_pred,average='macro'))
print("\n")





