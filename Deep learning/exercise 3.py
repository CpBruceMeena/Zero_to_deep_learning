import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("F://PycharmProjects//Zero_to_deep_learning//Deep learning//diabetes.csv")
print(df.describe())
print(df.head())

from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

ss = StandardScaler()
X = ss.fit_transform(df.drop('Outcome',axis = 1))
y_true = df[['Outcome']].values

y_cat = to_categorical(y_true)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size= 0.2, random_state=22)

print(X.shape, y_cat.shape)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import  GaussianNB

#the below line is common for all the algorithms
y_test_class = np.argmax(y_test, axis = 1)

print('Initially we are using decision tree classifier')
d_tree_model = DecisionTreeClassifier(max_depth= 5).fit(X_train, y_train)
y_pred_d = d_tree_model.predict(X_test)

y_pred_class = np.argmax(y_pred_d, axis = 1)

print(classification_report(y_pred_class,y_test_class))
print(confusion_matrix(y_pred_class,y_test_class))
print("the accuracy score for the decision tree classifier is {:.3f}".format(accuracy_score(y_pred_class, y_test_class)))
print()

print("here we are using random forest classifier")
# If we increase the max_depth of the random forest classfier then the acucracy increases
random_f = RandomForestClassifier(n_estimators= 10, max_depth= 5).fit(X_train, y_train)
y_pred_f = random_f.predict(X_test)

y_pred_class = np.argmax(y_pred_f, axis = 1)

print(classification_report(y_pred_class,y_test_class))
print(confusion_matrix(y_pred_class,y_test_class))
print("the accuracy score for the random tree classifier is {:.3f}".format(accuracy_score(y_pred_class, y_test_class)))

print()

#for using SVM, you need to make sure that the y_train has only one dimension
# and we cannot use np.argmax on the predicted results of the SVM model
print("here we are using Support Vector Machines")
model = SVC(kernel='linear', C =1.0, gamma = 'auto')

model.fit(X_train, y_train[:, 1])

y_pred_svc = model.predict(X_test)

print(classification_report(y_pred_svc,y_test_class))
print(confusion_matrix(y_pred_svc,y_test_class))
print("the accuracy score for the SVM is {:.3f}".format(accuracy_score(y_pred_svc, y_test_class)))

print("Here we are using Naive Byes")

G_model = GaussianNB().fit(X_train, y_train[:, 1])
y_pred_d = G_model.predict(X_test)

print(classification_report(y_pred_d,y_test_class))
print(confusion_matrix(y_pred_d,y_test_class))
print("the accuracy score for the Naive Byes is {:.3f}".format(accuracy_score(y_pred_d, y_test_class)))

print()
