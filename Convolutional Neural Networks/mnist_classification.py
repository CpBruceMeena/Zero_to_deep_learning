import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data("F://PycharmProjects//Zero_to_deep_learning//Convolutional Neural Networks//MNIST.npz")
print(X_train.shape)

#plt.imshow(X_train[2], cmap = 'gray')

#below -1 is used to convert it into grayscale
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train/255.0
X_test = X_test/255.0

from keras.utils import to_categorical
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

from keras.models import Sequential
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers import Dense

import keras.backend as K
K.clear_session()

model = Sequential()
model.add(Dense(512, input_shape=(28*28, ), activation='relu'))
model.add(Dense(256, activation = 'sigmoid'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation = 'softmax'))

model.compile(RMSprop(lr = 0.01), loss = 'categorical_crossentropy', metrics = ['accuracy'])
h = model.fit(X_train, y_train, verbose =1, epochs = 10, validation_split= 0.1, batch_size=128)

plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.legend(['Training', 'Validation'])
plt.title('Accuracy')
plt.xlabel('Epoch')

test_accuracy = model.evaluate(X_test, y_test)[1]
plt.show()
