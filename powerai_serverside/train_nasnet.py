from keras import backend as K
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.layers import Dense
from keras.applications.nasnet import NASNetMobile
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from datasequence import DataSequence
import pandas as pd
import sys

def euc_dist_keras(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))

model = NASNetMobile()
model = Model(model.input, Dense(1, activation='linear', kernel_initializer='normal')(model.layers[-2].output))
model.compile("adam", euc_dist_keras)
model.summary()

train_pd = pd.read_csv("splits/train_1.txt")
test_pd = pd.read_csv("splits/test_1.txt")
batch_size = 32

lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.003)
tb_callback = TensorBoard()
mc_callback = ModelCheckpoint(filepath='current_best.hdf5', verbose=1, save_best_only=True)

model.fit_generator(DataSequence(train_pd, "./images", batch_size=batch_size), validation_data=DataSequence(test_pd, "./images", batch_size=batch_size), epochs=20, verbose=1, use_multiprocessing=True, workers=80, steps_per_epoch=len(train_pd) // batch_size, validation_steps=len(test_pd) // batch_size, callbacks=[lr_callback, tb_callback, mc_callback])

model.save("nasnet_lamem_model.h5")
