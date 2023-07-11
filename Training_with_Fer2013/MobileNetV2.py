
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential 
from keras.layers import Dense,Dropout,Flatten, BatchNormalization, GlobalAveragePooling2D
from tensorflow.python.lib.io import file_io
#from keras.layers import Conv2D, Maxpooling2D
from keras.applications import VGG19
from keras_vggface.vggface import VGGFace
from tensorflow.keras.optimizers.legacy import Adam

import os 
from matplotlib import pyplot as plt
import numpy as np
from glob import glob
from tensorflow.keras import layers, Model
from keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.models import load_model
import keras.backend as K
from keras.utils import plot_model
from sklearn.metrics import *
from tensorflow.keras import Model
import skimage
from skimage.transform import rescale, resize

import pydot

random.seed(42)
tf.random.set_seed(42)

train_data_dir ="Archive/Train" #FER2013 dataset train set
validation_data_dir ="Archive/test-public" #fer2013 datset test set


model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(224, 224, 3), pooling='avg', weights = 'imagenet')
base_input = model.layers[0].input
base_output = model.layers[-3].output
model = Model(base_input, base_output)
model.summary()



x = GlobalAveragePooling2D()(base_output)
x = Dense(1024, activation='relu')(x)
out = Dense(7, activation='softmax', name='classifier')(x)#7 classes in FER2013
model = Model(base_input, out)


sgd = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
filepath="VGG16_weights_improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', verbose=1)

callbacks_list = [checkpoint, reduce_lr, early_stop]
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])



print(model.summary())


def get_datagen(dataset, aug=False):
    if aug:
        datagen = ImageDataGenerator(
                            rescale=1./255,
                            featurewise_center=False,
                            featurewise_std_normalization=False,
                            rotation_range=10, 
                            width_shift_range=0.1, #float: fraction of total width, if < 1, or pixels if >= 1.
                            height_shift_range=0.1,
                            zoom_range=0.1,
                            horizontal_flip=True)  
    else:
        datagen = ImageDataGenerator(rescale=1./255)

    return datagen.flow_from_directory(
            dataset,
            color_mode='rgb',
            target_size=(224, 224), 
            shuffle = True,
            class_mode='categorical',
            batch_size=32)




Train_Data_Generator = get_datagen('Archive/Train', True)
Validation_Data_Generator = get_datagen('Archive/test-public')
test_datagen  = get_datagen('Archive/test-private')


batch_size=32
history = model.fit(
    Train_Data_Generator,
    validation_data = Validation_Data_Generator, 
    steps_per_epoch=28709// batch_size,
    validation_steps=3509 // batch_size,
    shuffle=True,
    epochs=50,
    callbacks=[callbacks_list],
) 



# Load the weights of the best validation accuracy model
model.load_weights('MobileNetv2_weights_improvement-29-0.69.hdf5')

# Compile the model with the same optimizer, loss function, and metrics as during training
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(Validation_Data_Generator)
print('Validation loss:', val_loss)
print('Validation accuracy:', val_accuracy)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_datagen)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)


# Evaluate the model on the test set
train_loss, train_accuracy = model.evaluate(Train_Data_Generator)
print('Test loss:', train_loss)
print('Test accuracy:', train_accuracy)


batch_size=32
print('\n# Evaluate on test data')
results_test = model.evaluate_generator(test_datagen, 3509 // batch_size)
print('test loss, test accuracy:', results_test)


plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()


plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()
