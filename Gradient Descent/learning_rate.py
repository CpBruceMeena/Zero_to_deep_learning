import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("F://PycharmProjects//Zero_to_deep_learning//Gradient Descent//bank_notes.csv")
print(df.describe())
print(df.head())

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

X = scale(df.drop(['class'], axis = 1).values)
y_true = df[['class']].values

print(X.shape, y_true.shape)
'''
import seaborn as sns
sns.pairplot(df, hue = 'class')
plt.show()
'''

from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.layers import Dense

model = RandomForestClassifier()
cross_val_score(model, X, y_true)

X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size= 0.3, random_state=42)

import keras.backend as K

model = Sequential()
model.add(Dense(1, input_shape=(4, ), activation= 'sigmoid'))
model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs = 30, verbose = 2, validation_split= 0.1)

#Evaluate gives the accuracy and loss, while the model.predict gives the prediction that is basically the output for the given input
#model.evaluate gives the loss and accuracy for 0 and 1 index respectively
result = model.evaluate(X_test, y_test)

history = pd.DataFrame(history.history, index = history.epoch)
history.plot(ylim = (0, 1))
plt.title('the accuracy for the test set is {:.3f}'.format(result[1]*100), fontsize = 15)

dflist = []
learning_rates = [0.01, 0.05, 0.1, 0.5]

for lr in learning_rates:
        K.clear_session()

        model = Sequential()
        model.add(Dense(1, input_shape=(4, ), activation= 'sigmoid'))
        model.compile(Adam(lr = lr), loss = 'binary_crossentropy', metrics=['accuracy'])

        h = model.fit(X_train, y_train, verbose = 2, batch_size = 16)
        dflist.append(pd.DataFrame(h.history, index = h.epoch))


historydf = pd.concat(dflist, axis = 1)

metrics_reported = dflist[0].columns
idx = pd.MultiIndex.from_product([learning_rates, metrics_reported], names = ['learning_rate', 'metric'])
historydf.columns = idx

ax = plt.subplot(211)
historydf.xs('loss', axis = 1, level = 'metric').plot(ylim = (0,1), ax = ax)
plt.title('loss')

ax = plt.subplot(212)
historydf.xs('acc', axis = 1, level = 'metric').plot(ylim = (0, 1), ax = ax)
plt.title('Accuracy')
plt.xlabel('epochs')
plt.tight_layout()

plt.show()
