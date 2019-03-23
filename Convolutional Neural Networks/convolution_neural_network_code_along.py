import matplotlib.pyplot as plt
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data("F://PycharmProjects//Zero_to_deep_learning//Convolutional Neural Networks//MNIST.npz")

#plt.imshow(X_train[2], cmap = 'gray')
#below -1 is used to convert it into grayscale
X_train = X_train.reshape(-1, 28, 28, 1)  #dont forget to reshape the image
X_test = X_test.reshape(-1, 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train/255.0 #This is very important as it normalizes the image parameters
X_test = X_test/255.0

from keras.utils import to_categorical
y_train_cat = to_categorical(y_train) #this helps us to create the sparse matrix
y_test_cat = to_categorical(y_test)

from keras.layers import Conv2D,AvgPool2D, MaxPool2D, Activation, Flatten
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
import keras.backend as K
K.clear_session()

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

print(model.summary())

model.fit(X_train, y_train_cat, verbose = 2, epochs = 2, batch_size=128, validation_split = 0.3)
result = model.evaluate(X_test, y_test_cat)
print(result)