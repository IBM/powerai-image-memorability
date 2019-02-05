import coremltools
import sys
from tensorflow.keras.models import load_model

from keras.models import Sequential
from keras.layers import *

tf_model = load_model(sys.argv[1])
weights = tf_model.get_weights()

model = Sequential()
model.add(Conv2D(96, (11, 11), strides=(4, 4), activation="relu", input_shape=(227, 227, 3)))
model.add(MaxPooling2D((3, 3), (2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(256, (5, 5), activation="relu"))
model.add(ZeroPadding2D((2, 2)))
model.add(MaxPooling2D((3, 3), (2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(384, (3, 3), activation="relu"))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(384, (3, 3), activation="relu"))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation="relu"))
model.add(ZeroPadding2D((1, 1)))
model.add(MaxPooling2D((3, 3), (2, 2)))
model.add(GlobalAveragePooling2D())
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1))
model.set_weights(weights)

coreml_model = coremltools.converters.keras.convert(model, input_names="data", image_input_names="data", image_scale=1./255.)
coreml_model.save("lamem.mlmodel")
