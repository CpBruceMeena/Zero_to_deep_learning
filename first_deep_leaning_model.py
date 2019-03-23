import numpy as np
from sklearn.datasets.samples_generator import make_circles
from keras.callbacks import EarlyStopping

X, y = make_circles(n_samples=4000, noise=0.1, factor = 0.2, shuffle=True, random_state= 0)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

model = Sequential()
model.add(Dense(4, input_shape = (2, ), activation = 'tanh'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))


es = EarlyStopping(verbose = 1, patience= 3)

model.compile(Adam(lr=0.1), 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X,y, epochs = 20, callbacks=[es])
