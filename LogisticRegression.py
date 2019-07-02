import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import svm
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('car.data')

#print("Class Distribution (number of instances per class)\n")
#print("class      N          N[%]")
#print("unacc     1210     (70.023 %) ")
#print("acc        384     (22.222 %) ")
#print("good        69     ( 3.993 %) ")
#print("v-good      65     ( 3.762 %) ")

print("\n")

data.columns = ["buying", "maint", "doors", "persons","lug_boot","safety","class"]

le = preprocessing.LabelEncoder()
data_new= data.apply(le.fit_transform)
data_y = data_new['class']
#print(data_y)
del data_new['class']
data_x = data_new

#print(data_x)

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=2, stratify=data_y)

#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)

clf = LogisticRegression(solver='newton-cg',multi_class='auto')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn import metrics
print("Logistic Regression")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred,average='macro'))
print("Precision:",metrics.precision_score(y_test, y_pred,average='macro'))

#from sklearn.metrics import classification_report
#print(classification_report(y_test,y_pred))

print("\n")
