import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("F://PycharmProjects//Zero_to_deep_learning//Deep learning//iris.csv")
print(df.describe())
print(df.head())

X = df.drop('species', axis = 1)
target_name = df['species'].unique()

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD

target_dict = {n:i for i, n in enumerate(target_name)}

y = df['species'].map(target_dict)
from keras.utils.np_utils import to_categorical

y_cat = to_categorical(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2)

model = Sequential()
model.add(Dense(3, input_shape=(4, ), activation= 'softmax'))
model.compile(Adam(lr = 0.3), loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, verbose = 2, epochs = 20, validation_split = 0.1)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

y_test_class = np.argmax(y_test, axis = 1)
y_pred_class = np.argmax(y_pred, axis = 1)

#print('the accuracy on the test score is {:.3f}'.format(accuracy_score(y_test, y_pred_class)))
print(y_pred_class.shape, y_test.shape, y_test_class.shape)

from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test_class, y_pred_class))
print(confusion_matrix(y_test_class, y_pred_class))


