import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("F://PycharmProjects//Zero_to_deep_learning//user_visiting_duration.csv")

#print(df.head(10))

X = df[['Time (min)']].values
y_true = df[['Buy']].values

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_shape=(1, ), activation='sigmoid'))
model.compile(optimizer=Adam(lr = 0.4), loss= 'binary_crossentropy', metrics=['accuracy'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2)
model.fit(X_train, y_train, epochs = 40, verbose = 2)

y_train_pred = model.predict(X_train).ravel()
y_test_pred = model.predict(X_test).ravel()

df.plot(kind = 'scatter', x = 'Time (min)', y = 'Buy')

from sklearn.metrics import mean_squared_error as mse

temp = np.linspace(0, 4)
plt.plot(temp, model.predict(temp), color = 'red')
#plt.plot(X_test, y_test_pred, color = 'red')

plt.plot(temp, model.predict(temp) > 0.5, color = 'blue')

print('The mse on the train set is {:.3f}'.format(mse(y_train, y_train_pred)))
print('The mse on the test set is {:.3f}'.format(mse(y_test, y_test_pred)))

plt.show()