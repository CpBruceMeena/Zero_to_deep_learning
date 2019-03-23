import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

print(X.shape)
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.model_selection import learning_curve
import keras.backend as K

K.clear_session()

y_cat = to_categorical(y, 10)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size= 0.3)
model = Sequential()

model.add(Dense(16, input_shape=(64, ), activation= 'relu'))
model.add(Dense(10, activation= 'softmax'))
model.compile(loss = 'categorical_crossentropy', metrics=['accuracy'], optimizer= 'adam')

initial_weights = model.get_weights()

train_sizes = (len(X_train)*np.linspace(0.1, 0.999, 4)).astype(int)

train_scores = []
test_scores = []

for train_size in train_sizes:
    X_train_fac, _, y_train_fac, _ = train_test_split(X_train, y_train, train_size = train_size)

    model.set_weights(initial_weights)

    h = model.fit(X_train_fac, y_train_fac, epochs = 300, verbose = 1, callbacks=[EarlyStopping(patience= 1, verbose = 2, monitor = 'loss')])

    r = model.evaluate(X_train_fac, y_train_fac, verbose = 0)
    train_scores.append(r[-1])

    s = model.evaluate(X_test, y_test, verbose = 0)
    test_scores.append(s[-1])

    print('test size done:', train_size)

plt.plot(train_sizes, train_scores, 'o-', label = 'training score')
plt.plot(train_sizes, test_scores, 'o-', label = 'test score')

plt.show()