
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation,BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Model

def get_callbacks_list():
    return [callbacks.EarlyStopping(monitor='val_accuracy',patience=25)]

def VGG16(Input_shape=(224,224,3),No_classes=4):
    _input = Input(Input_shape) 

    Conv1  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(_input)
    Conv2  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(Conv1)
    pool1  = MaxPooling2D((2, 2))(Conv2)

    Conv3  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(pool1)
    Conv4  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(Conv3)
    pool2  = MaxPooling2D((2, 2))(Conv4)

    Conv5  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(pool2)
    Conv6  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(Conv5)
    Conv7  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(Conv6)
    pool3  = MaxPooling2D((2, 2))(Conv7)

    Conv8  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool3)
    Conv9  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(Conv8)
    Conv10 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(Conv9)
    pool4  = MaxPooling2D((2, 2))(Conv10)

    Conv11 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool4)
    Conv12 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(Conv11)
    Conv13 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(Conv12)
    pool5  = MaxPooling2D((2, 2))(Conv13)

    flat   = Flatten()(pool5)
    dense1 = Dense(4096, activation="relu")(flat)
    dense2 = Dense(4096, activation="relu")(dense1)
    output = Dense(No_classes, activation="softmax")(dense2)

    vgg16_model  = Model(inputs=_input, outputs=output)
    return vgg16_model

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory("images",
                                              target_size=(128,128),
                                              batch_size=8,
                                              class_mode='categorical',
                                              shuffle = True,seed=1)

Classes=list(train_generator.class_indices.keys())

VGG16_Model=VGG16(Input_shape=(128, 128,3),No_classes=len(Classes))
VGG16_Model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

train_steps = len(train_generator.filenames) // 8
history = VGG16_Model.fit_generator(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=100,
    callbacks=get_callbacks_list())

VGG16_Model.save('VGG16_Model.h5')