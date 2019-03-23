from keras.layers import BatchNormalization, Dense
from keras.models import Sequential
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

def repeated_training(X_train, y_train, X_test, y_test, units = 512,
                      activation = 'sigmoid', optimizers = 'sgd',
                      do_bn = False, epochs = 10, repeats = 3):

    histories = []

    for i in range(repeats):
        K.clear_session()

        model = Sequential()

        #first fully connected layer
        model.add(Dense(units, input_shape = X_train.shape[1:],
                        kernel_initializer= 'normal',
                        activation = activation))

        if do_bn:
            model.add(BatchNormalization())

        #second fully connected layer
        model.add(Dense(units, kernel_initializer='normal', activation=activation))

        if do_bn:
            model.add(BatchNormalization())

        #third fully connected layer
        model.add(Dense(units, kernel_initializer='normal', activation=activation))

        if do_bn:
            model.add(BatchNormalization())

        #Output
        model.add(Dense(10, activation = 'softmax'))

        #Never involve verbose in model.compile
        model.compile(optimizers, loss = 'categorical_crossentropy', metrics= ['accuracy'])

        h = model.fit(X_train, y_train, validation_data= (X_test, y_test), epochs = epochs, verbose = 2)

        histories.append([h.history['acc'], h.history['val_acc']])
        print(i, end = ' ')

    histories = np.array(histories)

    mean_acc = histories.mean(axis = 0)
    std_acc = histories.std(axis = 0)

    print()

    return mean_acc[0], std_acc[0], mean_acc[1], std_acc[1]

from sklearn.datasets import load_digits
from keras.utils import to_categorical

digits = load_digits()
X, y = digits.data, digits.target

y_cat = to_categorical(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size= 0.3)

mean_acc, std_acc, mean_acc_val, std_acc_val = repeated_training(X_train, y_train, X_test, y_test, do_bn = False)
mean_acc_bn, std_acc_bn, mean_acc_val_bn, std_acc_val_bn = repeated_training(X_train, y_train, X_test, y_test, do_bn = True)

def plot_mean_std(m, s):
    plt.plot(m)
    plt.fill_between(range(len(m)), m-s, m+s, alpha = 0.1)

plot_mean_std(mean_acc, std_acc)
plot_mean_std(mean_acc_val, std_acc_val)
plot_mean_std(mean_acc_bn, std_acc_bn)
plot_mean_std(mean_acc_val_bn, std_acc_val_bn)
plt.ylim(0, 1.01)
plt.title('Batch Normalization Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test', 'Train with Batch Normalization', 'Test with Batch Normalization'])

plt.show()