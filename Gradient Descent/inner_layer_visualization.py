import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("F://PycharmProjects//Zero_to_deep_learning//Gradient Descent//bank_notes.csv")
print(df.describe())
print(df.head())

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

X = scale(df.drop(['class'], axis = 1).values)
y_true = df[['class']].values

from keras.models import Sequential
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Dense
import keras.backend as K

X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size= 0.2, random_state=42)

import keras.backend as K
'''
model = Sequential()
model.add(Dense(2, input_shape = (4, ), activation = 'relu'))
model.add(Dense(1, activation= 'sigmoid'))
model.compile(RMSprop(lr = 0.01), loss = 'binary_crossentropy', metrics=['accuracy'])

h = model.fit(X_train, y_train, verbose = 2, batch_size=16, validation_split=0.1, epochs = 20)
result = model.evaluate(X_test, y_test)

inp = model.layers[0].input
out = model.layers[0].output

features_function = K.function([inp], [out])
features = features_function([X_test])[0]
plt.scatter(features[:, 0], features[:, 1], c= y_test, cmap = 'coolwarm')
'''
K.clear_session()

model = Sequential()
model.add(Dense(3, input_shape = (4, ), activation = 'relu'))
model.add(Dense(2, activation = 'relu'))
model.add(Dense(1, activation= 'sigmoid'))
model.compile(RMSprop(lr = 0.01), loss = 'binary_crossentropy', metrics=['accuracy'])

inp = model.layers[0].input
out = model.layers[1].output

features_function = K.function([inp], [out])
plt.figure(figsize=(15, 10))

for i in range(1, 26):
    plt.subplot(5, 5, i)

    h = model.fit(X_train, y_train, verbose=2, batch_size=16, epochs=1)
    test_accuracy = model.evaluate(X_test, y_test)[1]
    features = features_function([X_test])[0]
    plt.scatter(features[:, 0], features[:, 1], c=y_test)
    plt.xlim(-0.5, 3.5)
    plt.ylim(-0.5, 4.0)
    plt.title("Epoch:{}, Test Acc = {:3.1}%".format(i, test_accuracy * 100))
plt.tight_layout()
plt.show()
