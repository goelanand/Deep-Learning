import tensorflow as tf
import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import cv2
import matplotlib.pyplot as plt


img = image.load_img('color_dataset/womens hat3.jpg')


img =img_to_array(img)

print(img.shape)

datagen = image.ImageDataGenerator(
    featurewise_center=True, samplewise_center=True,
    featurewise_std_normalization=True, samplewise_std_normalization=True,
    zca_whitening=True, zca_epsilon=1e-06, rotation_range=40, width_shift_range=0.2,
    height_shift_range=0.20, brightness_range=(0.1,0.3), shear_range=0.2, zoom_range=0.2,
    channel_shift_range=0.20, fill_mode='nearest', cval=0.10, rescale=None,
    horizontal_flip=True, vertical_flip=True, preprocessing_function=None,
    data_format=None, validation_split=0.20, dtype=None, )

img = img.reshape((1,) + img.shape)
print(img.shape)
i = 0

for x in datagen.flow(img, batch_size=1, save_to_dir='bin/e/', save_prefix='Pics',
                      save_format='jpg'):
    i += 1
    if i > 30:
        break

print('')