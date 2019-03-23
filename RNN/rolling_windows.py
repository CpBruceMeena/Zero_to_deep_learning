import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("F://PycharmProjects//Zero_to_deep_learning//RNN//cansim_eng.csv")

from pandas.tseries.offsets import MonthEnd

df['Adjustments'] = pd.to_datetime(df['Adjustments']) + MonthEnd(1)
df = df.set_index('Adjustments')

split_date = pd.Timestamp('01-01-2011')
train = df.loc[:split_date, ['Unadjusted']]
test = df.loc[split_date:, ['Unadjusted']]

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
train_sc = sc.fit_transform(train)
test_sc = sc.transform(test)

X_train = train_sc[:-1]
y_train = train_sc[1:]

X_test = test_sc[:-1]
y_test = test_sc[1:]

#Here we are using window rolling
train_sc_df = pd.DataFrame(train_sc, columns = ['Scaled'], index = train.index)
test_sc_df = pd.DataFrame(test_sc, columns = ['Scaled'], index = test.index)

for s in range(1, 13):
    train_sc_df['shift_{}'.format(s)] = train_sc_df['Scaled'].shift(s)
    test_sc_df['shift_{}'.format(s)] = test_sc_df['Scaled'].shift(s)

X_train = train_sc_df.dropna().drop('Scaled', axis = 1)
y_train = train_sc_df.dropna()[['Scaled']]

X_test = test_sc_df.dropna().drop('Scaled', axis = 1)
y_test = test_sc_df.dropna()[['Scaled']]

X_train = X_train.values
y_train = y_train.values

X_test = X_test.values
y_test = y_test.values

import keras.backend as K
K.clear_session()
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(Dense(12, input_dim = 12, activation= 'relu'))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
print(model.summary())

early_stop = EarlyStopping(patience = 1, verbose = 2, monitor = 'loss')
model.fit(X_train , y_train, epochs = 200, verbose = 1, callbacks=[early_stop], batch_size= 1)
y_pred = model.predict(X_test)

#plt.plot(y_pred)
#plt.plot(y_test)

#Now we are implementing LSTM
#for lstm we need the input with 3D tensor with shape(batch_size , timesteps, input_dim)
from keras.layers import LSTM

X_train_t = X_train.reshape(X_train.shape[0], 1, 12)
X_test_t = X_test.reshape(X_test.shape[0], 1, 12)

K.clear_session()
model = Sequential()
model.add(LSTM(6, input_shape=(1, 12)))
model.add(Dense(1))

model.compile(optimizer= 'adam',loss  = 'mean_squared_error')
model.fit(X_train_t, y_train, epochs = 100, batch_size = 1, verbose = 1, callbacks=[early_stop])

y_pred = model.predict(X_test_t)
plt.plot(y_pred)
plt.plot(y_test)

plt.show()