import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Activation, Dense, Conv2D, MaxPool2D, Flatten
from keras.optimizers import rmsprop

import keras.backend as K
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data("F://PycharmProjects//Zero_to_deep_learning//Convolutional Neural Networks//MNIST.npz")

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train/255.0
X_test = X_test/255.0

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

model = Sequential()

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation= 'relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Activation('relu'))

model.add(Dense(128, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics = ['accuracy'])

print(model.summary())
model.fit(X_train, y_train_cat, verbose = 1, epochs = 5, validation_split=0.3, batch_size=128)

