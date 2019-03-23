import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

plt.imshow(X_train[9])
plt.show()
print(X_train.shape)

X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0

from keras.utils import to_categorical
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

import keras.backend as K
K.clear_session()

from keras.models import Sequential
from keras.layers import Activation, Dense, Conv2D, MaxPool2D, Flatten
from keras.optimizers import RMSprop

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape= (32, 32, 3),padding= 'same', activation= 'relu'))
model.add(Conv2D(64, (3, 3), padding= 'same',activation='relu'))

model.add(MaxPool2D(pool_size= (2, 2)))

model.add(Conv2D(128, (3, 3), activation= 'relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPool2D(pool_size= (2, 2)))

model.add(Flatten())
model.add(Dense(256, activation= 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer= 'rmsprop')
print(model.summary())

model.fit(X_train, y_train_cat, verbose = 1, epochs = 2, batch_size= 128, validation_split=0.3)

result = model.evaluate(X_test, y_test_cat)
print('the loss on the testing set is {:.3f}'.format(result[0]))
print('the accuracy on the testing set is {:.3f}'.format(result[1]))
