import numpy as np
from scipy.misc import ascent
from keras.layers import Conv2D
import matplotlib.pyplot as plt

img = ascent()
print(img.shape)
#plt.imshow(img, cmap ='gray')

img_tensor = img.reshape((1, 512, 512, 1))

from keras.layers import MaxPool2D,Conv2D, AvgPool2D

from keras.models import Sequential
'''
model = Sequential()
model.add(Conv2D(1, (3, 3), input_shape=(512, 512, 1), strides = (1, 1)))
model.add(MaxPool2D((5, 5), strides = (1, 1)))
model.compile('adam', 'mse')
img_pred = model.predict(img_tensor)

print(img_pred.shape)
plt.imshow(img_pred[0, :, :, 0], cmap = 'gray')
'''
model = Sequential()
model.add(AvgPool2D((3, 3), strides = (1, 1)))
model.compile('adam', 'mse')
img_pred = model.predict(img_tensor)
plt.imshow(img_pred[0, :, :, 0], cmap = 'gray')
plt.show()
print(img_pred.shape)