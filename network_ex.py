import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

images = np.array([])
meas = np.array([])
goal = np.array([])


# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

image_input = keras.layers.Input(shape=(84,84,1))
f1 = Conv2D(64, (3, 3), activation='relu')(image_input)
f2 = 

# input1 = keras.layers.Input(shape=(16,))
# x1 = keras.layers.Dense(8, activation='relu')(input1)
# input2 = keras.layers.Input(shape=(32,))
# x2 = keras.layers.Dense(8, activation='relu')(input2)
# added = keras.layers.Add()([x1, x2])  # equivalent to added = keras.layers.add([x1, x2])

# out = keras.layers.Dense(4)(added)
# model = keras.models.Model(inputs=[input1, input2], outputs=out)