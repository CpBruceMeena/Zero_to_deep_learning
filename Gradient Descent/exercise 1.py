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

#import seaborn as sns
#sns.pairplot(df)

ss = StandardScaler()

X = ss.fit_transform(X)

from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size= 0.2, random_state= 42)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam, RMSprop

model = Sequential()
model.add(Dense(8, input_shape=(13, ),kernel_initializer= 'he_normal' ,activation= 'sigmoid'))
model.add(Dense(5,kernel_initializer= 'he_normal', activation= 'tanh'))
model.add(Dense(2,kernel_initializer= 'he_normal', activation = 'tanh'))
model.add(Dense(3, activation = 'softmax'))

model.compile(RMSprop(lr = 0.05), loss = 'categorical_crossentropy', metrics= ['accuracy'])
model.fit(X, y_cat, verbose = 2, epochs = 20, batch_size= 16)

inp = model.layers[0].input
out = model.layers[2].output

import keras.backend as K
features_function = K.function([inp], [out])
features = features_function([X])[0]
plt.scatter(features[:, 0], features[:, 1], c = y_cat)

plt.show()
