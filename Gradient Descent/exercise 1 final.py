import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("F://PycharmProjects//Zero_to_deep_learning//Gradient Descent//wines.csv")
print(df.describe())
print(df.head())

from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale

X = scale(df.drop(['Class'], axis = 1).values)
y = df['Class']

print(y.value_counts())
#To convert the single values of y in a two dimension sparse matrix
y_cat = pd.get_dummies(y)

y_cat = y_cat.values

ss = StandardScaler()

X = ss.fit_transform(X)

from keras.models import Model
from keras.layers import Input, Dense
import keras.backend as K
from keras.optimizers import RMSprop, Adam

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.3, random_state = 42)

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

checkpoint = ModelCheckpoint(filepath = "F://PycharmProjects//Zero_to_deep_learning//Gradient Descent//weights.hdf5", verbose = 1, save_best_only=True)

earlyStopper = EarlyStopping(monitor = 'val_loss', min_delta= 0, patience = 2, verbose = 1, mode = 'auto')
tensorboard = TensorBoard(log_dir='F://PycharmProjects//Zero_to_deep_learning//Gradient Descent//tensorboard' )

inputs = Input(shape = (13, ))
x = Dense(8, kernel_initializer= 'he_normal', activation = 'tanh')(inputs)
x = Dense(5, kernel_initializer= 'he_normal', activation = 'tanh')(x)
second_to_last = Dense(2, kernel_initializer= 'he_normal', activation = 'tanh')(x)
outputs = Dense(3, activation = 'softmax')(second_to_last)

model = Model(inputs = inputs, outputs = outputs)

model.compile(RMSprop(lr = 0.01), loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, verbose = 2, epochs = 20, batch_size = 16, validation_data=(X_test, y_test), callbacks=[checkpoint, earlyStopper, tensorboard])
