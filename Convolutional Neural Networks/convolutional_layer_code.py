import numpy as np
from scipy.misc import ascent
from keras.layers import Conv2D
import matplotlib.pyplot as plt

img = ascent()
print(img.shape)
#plt.imshow(img, cmap ='gray')

img_tensor = img.reshape((1, 512, 512, 1))
from keras.models import Sequential
from keras.optimizers import  RMSprop

model = Sequential()
model.add(Conv2D(1, (3, 3), strides = (1, 1),input_shape=(512, 512, 1)))
model.compile('adam', 'mse')

img_pred_tensor = model.predict(img_tensor)
print(img_pred_tensor.shape)

img_pred = img_pred_tensor[0, :, :, 0]
#plt.imshow(img_pred, cmap ='gray')
weights = model.get_weights()
#plt.imshow(weights[0][:, :, 0, 0], cmap = 'gray')

weights[0] = np.ones(weights[0].shape)
model.set_weights(weights)
img_pred_tensor = model.predict(img_tensor)

plt.imshow(img_pred_tensor[0,:, :, 0], cmap = 'gray')
plt.show()