import pandas as pd
import matplotlib.pyplot as plt
import csv

df = pd.read_csv("F://PycharmProjects//Zero_to_deep_learning//Deep learning//diabetes.csv")
print(df.describe())
print(df.head())

with open("F://PycharmProjects//Zero_to_deep_learning//Deep learning//diabetes.csv", "r") as f:
    d_reader = csv.DictReader(f)
    headers = d_reader.fieldnames
    print(headers)

#pd.plotting.scatter_matrix(df)
#df.hist()

'''
import seaborn as sns
fig, ax = plt.subplots(1, 2)
g = sns.pairplot(df, hue = 'Outcome')
h = sns.heatmap(df.corr(), annot = True, ax=ax[1])
fig.show()
'''
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

ss = StandardScaler()
X = ss.fit_transform(df.drop('Outcome',axis = 1))
y_true = df[['Outcome']].values

#We use to categorical to convert the single digit values in tersm of matrix type notation so that at the end
#we can get the probabilistic values of the diabeties

y_cat = to_categorical(y_true)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size= 0.2, random_state=22)

print(X.shape, y_cat.shape)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import  Adam, SGD

model = Sequential()
model.add(Dense(32, input_shape = (8, ), activation = 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(2, activation = 'softmax'))

model.compile(Adam(lr = 0.03), loss = 'categorical_crossentropy', metrics= ['accuracy'])
model.fit(X_train, y_train, verbose= 2, epochs= 20, validation_split=0.1)

print(model.summary())

import numpy as np

y_pred = model.predict(X_test)
y_test_class = np.argmax(y_test, axis = 1)
y_pred_class = np.argmax(y_pred, axis = 1)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print(accuracy_score(y_test_class, y_pred_class))
print(classification_report(y_test_class, y_pred_class))
print(confusion_matrix(y_test_class, y_pred_class))

plt.show()

