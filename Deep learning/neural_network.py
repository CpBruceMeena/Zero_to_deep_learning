import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

X, y = make_moons(n_samples= 1000, noise = 0.2, random_state = 0)
from sklearn.model_selection import train_test_split

print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD

model = Sequential()
model.add(Dense(8, input_shape=(2, ), activation='tanh'))
model.add(Dense(4, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(lr = 0.05), loss= 'binary_crossentropy' ,metrics = ['accuracy'])

model.fit(X_train, y_train, epochs= 100)
results = model.evaluate(X_test, y_test)

y_test_pred = model.predict_classes(X_test)
y_train_pred = model.predict_classes(X_train)

from sklearn.metrics import confusion_matrix, accuracy_score

print('the accuracy score on the test set is{:.3f}'.format(accuracy_score(y_test, y_test_pred)))
print('the accuracy score on the train set is{:.3f}'.format(accuracy_score(y_train, y_train_pred)))
