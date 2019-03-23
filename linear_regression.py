#The basic outline is 1. First of all, we have to get the X and y value from the table
#Remember how to import the table values fromt the data frame, using double square brackets
#then we need to add the dense layer and then compile
#always remember in the layer we need to input the number of nodes, activation function, shape
#then we need to compile the model, for this we need the optimizer, loss function and learning rate and then we need to fit the model using epochs
#The last step is to predict the values using the X and the plotting the line on the scatter plot

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam, SGD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("F://PycharmProjects//Zero_to_deep_learning//weight-height.csv")

print(df.head(5))
X = df[['Height']].values
y_true = df[['Weight']].values


'''
y_pred = model.predict(X)
df.plot(kind = 'scatter', x = 'Height', y = 'Weight')
plt.plot(X, y_pred, color = 'red')
print(y_pred)
plt.show()
W, B = model.get_weights()
print(W, B)
'''

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2)

model = Sequential()
model.add(Dense(1,input_shape = (1, )))
model.compile(Adam(lr = 0.8), 'mean_squared_error')
model.fit(X_train, y_train, epochs= 50, verbose = 1)

y_train_pred = model.predict(X_train).ravel()
y_test_pred = model.predict(X_test).ravel()

from sklearn.metrics import mean_squared_error as mse

print("The mean squared error on the train set is {:0.3f}".format(mse(y_train, y_train_pred)))
print('The mean squared error on the test set is {:0.3f}'.format(mse(y_test, y_test_pred)))

