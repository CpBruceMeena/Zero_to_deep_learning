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

y_cat = pd.get_dummies(y)

y_cat = y_cat.values

ss = StandardScaler()

X = ss.fit_transform(X)

from keras.models import Model
from keras.layers import Input, Dense
import keras.backend as K
from keras.optimizers import RMSprop, Adam

K.clear_session()
inputs = Input(shape = (13, ))
x = Dense(8, kernel_initializer= 'he_normal', activation = 'tanh')(inputs)
x = Dense(5, kernel_initializer= 'he_normal', activation = 'tanh')(x)

second_to_last = Dense(2, kernel_initializer= 'he_normal', activation = 'tanh')(x)

outputs = Dense(3, activation = 'softmax')(second_to_last)

model = Model(inputs = inputs, outputs = outputs)

model.compile(RMSprop(lr = 0.01), loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(X, y_cat, verbose = 2, epochs = 20, batch_size = 16)

features_function = K.function([inputs], [second_to_last])
features = features_function([X])[0]
#below c is the values that we want to distinguish
plt.scatter(features[:, 0], features[:, 1], c = y_cat)

plt.show()