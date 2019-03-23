import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("F://PycharmProjects//Zero_to_deep_learning//RNN//cansim_eng.csv")

print(df.head())

from pandas.tseries.offsets import MonthEnd

df['Adjustments'] = pd.to_datetime(df['Adjustments']) + MonthEnd(1)
df = df.set_index('Adjustments')

print(df.head())

#df.plot()

split_date = pd.Timestamp('01-01-2011')
train = df.loc[:split_date, ['Unadjusted']]
test = df.loc[split_date:, ['Unadjusted']]
'''
ax = train.plot()
test.plot(ax = ax)
plt.legend(['train', 'test'])
'''
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
train_sc = sc.fit_transform(train)
test_sc = sc.transform(test)

X_train = train_sc[:-1]
y_train = train_sc[1:]

X_test = test_sc[:-1]
y_test = test_sc[1:]

from keras.layers import Dense
from keras.models import Sequential
import keras.backend as K
from keras.callbacks import EarlyStopping
K.clear_session()

model = Sequential()
model.add((Dense(12, input_dim= 1, activation = 'relu')))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(patience= 1, verbose = 2, monitor = 'loss')

model.fit(X_train, y_train, epochs = 50, verbose = 1, batch_size= 2, callbacks=[early_stop])

y_pred = model.predict(X_test)

'''
plt.plot(y_test)
plt.plot(y_pred)

'''
#Now we are implementing LSTM
#for lstm we need the input with 3D tensor with shape(batch_size , timesteps, input_dim)
from keras.layers import LSTM

X_train_t = X_train[:, None]
X_test_t = X_test[:, None]

K.clear_session()
model = Sequential()
model.add(LSTM(6, input_shape=(1, 1)))
model.add(Dense(1))

model.compile(optimizer= 'adam',loss  = 'mean_squared_error')
model.fit(X_train_t, y_train, epochs = 100, batch_size = 1, verbose = 1, callbacks=[early_stop])

y_pred = model.predict(X_test_t)
plt.plot(y_pred)
plt.plot(y_test)

plt.show()