from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

generator = ImageDataGenerator(rescale = 1.0/255,
                               rotation_range=  20,
                               width_shift_range= 0.3,
                               height_shift_range= 0.3,
                               shear_range= 0.3,
                               zoom_range = 0.3,
                               horizontal_flip= True)

#the path needs to be the subdirectory means the images should be inside a subfolder and the address should be given of the outer folder,
#not the inside folder
train = generator.flow_from_directory("F://PycharmProjects//Zero_to_deep_learning//Improving performance//images",
                                      target_size=(128, 128),
                                      batch_size = 32,
                                      class_mode='binary')

plt.figure(figsize = (12, 12))

for i in range(25):
    img, label = train.next()
    plt.subplot(5, 5, i+1)
    plt.imshow(img[0])

plt.show()