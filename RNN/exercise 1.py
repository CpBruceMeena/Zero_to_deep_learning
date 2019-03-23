import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("F://PycharmProjects//Zero_to_deep_learning//RNN//cansim_eng.csv")
print(df.head())

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

#Below section is for taking the values, for the training we are taking values from 0 to n-1
#while for y we are taking values from 1 to n
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

X_train_t = X_train.reshape(X_train.shape[0], 12, 1)
X_test_t = X_test.reshape(X_test.shape[0], 12, 1)

from keras.layers import Dense, LSTM
from keras.models import Sequential
import keras.backend as K
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(LSTM(6, input_shape = (12, 1)))
model.add(Dense(1))

model.compile(loss = 'mean_squared_error', optimizer='rmsprop')
earlystop = EarlyStopping(verbose= 1, patience= 1, monitor = 'loss')

model.fit(X_train_t, y_train, callbacks=[earlystop], verbose=1, epochs = 200, batch_size=1)

y_pred = model.predict(X_test_t)
plt.plot(y_test)
plt.plot(y_pred)

plt.show()