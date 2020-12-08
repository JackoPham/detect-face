import argparse
import os
from os.path import *

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator

DATA_PREFIX = 'data'
DATASET_PREFIX = 'dataset'


class Training:
    def __init__(self, batch_size: int, ep, directory: str):
        self.ep = ep
        self._directory = directory
        self._classes_num = 0
        self._dataset = ''  # dataset directory
        self._output_model = ''  # out model file
        self._batch_size = batch_size
        np.random.seed(3)

        self._train_datagen = ImageDataGenerator(rescale=1. / 255,
                                                 validation_split=0.2)  # set validation split

        self._train_generator: DirectoryIterator = None
        self._valid_generator: DirectoryIterator = None

    @property
    def train_generator(self) -> DirectoryIterator:
        if bool(self._train_generator):
            return self._train_generator

        self._train_generator = self._train_datagen.flow_from_directory(
            self.dataset,
            target_size=(24, 24),
            color_mode='grayscale',
            batch_size=self._batch_size,
            class_mode='categorical',
            subset='training',
        )
        return self._train_generator

    @property
    def valid_generator(self) -> DirectoryIterator:
        if bool(self._valid_generator):
            return self._valid_generator

        self._valid_generator = self._train_datagen.flow_from_directory(
            self.dataset,
            target_size=(24, 24),
            color_mode='grayscale',
            batch_size=self._batch_size,
            class_mode='categorical',
            subset='training',
        )
        return self._valid_generator

    @property
    def dataset(self) -> str:
        if bool(self._dataset):
            return self._dataset

        # determine dataset
        for _, dirs, _ in os.walk(self._directory):
            dataset = DATASET_PREFIX if DATASET_PREFIX in dirs else dirs[0]
            self._dataset = join(self._directory, dataset)
            return self._dataset

    @property
    def classes_num(self) -> int:
        if self._classes_num > 0:
            return self._classes_num

        # determine classes number
        for _, dirs, _ in os.walk(self.dataset):
            self._classes_num = len(dirs)
            return self._classes_num

    @property
    def output_model(self) -> str:
        if bool(self._output_model):
            return self._output_model

        # determine output model file
        [title, _] = splitext(basename(self._directory))
        self._output_model = join(self._directory,
                                  title.replace(DATA_PREFIX + '_', '') + '.h5')
        return self._output_model

    def makeModel(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(24, 24, 1)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.classes_num, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        return model

    def training(self):
        model = self.makeModel()
        cb_early_stopping = EarlyStopping(monitor='val_loss', patience=20)
        path_model = '{epoch:d}' + self.output_model
        cb_checkpoint = ModelCheckpoint(filepath=path_model, monitor='val_loss',
                                        verbose=1, save_best_only=True)

        history = model.fit_generator(
            generator=self.train_generator,
            steps_per_epoch=int(
                self.train_generator.n / self._batch_size),
            epochs=self.ep,
            validation_data=self.valid_generator,
            validation_steps=self.valid_generator.n // self._batch_size,
            callbacks=[cb_checkpoint])

        print("-- Evaluate --")
        scores = model.evaluate_generator(self.valid_generator, steps=5)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

        print("-- Predict --")
        output = model.predict_generator(self.valid_generator, steps=5)
        np.set_printoptions(
            formatter={'float': lambda x: "{0:0.3f}".format(x)})
        print(self.valid_generator.class_indices)

        model.save(self.output_model)

        self.graph(history)

    def graph(self, history):
        plt.figure(1)
        # summarize history for accuracy
        plt.subplot(211)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        # summarize history for loss
        plt.subplot(212)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--data', type=str,
                    default='data', help='data directory')
    ap.add_argument('-ds', '--dataset', type=str,
                    default=DATASET_PREFIX, help='dataset directory')
    ap.add_argument('-l', '--label', type=str,
                    default='label', help='label file')
    ap.add_argument('-o', '--output', type=str,
                    default='model', help='output model file')
    args = vars(ap.parse_args())

    # determine data path
    data_path = args['data']

    # for fname in os.listdir('data_thai_chars'):
    #     if fname.endswith('.labels'):
    #         with open('topsites.txt') as file:
    #         array = file.readlines()
    #         # do stuff on the file
    #         break

    trainer = Training(batch_size=512, ep=1000, directory=data_path)
    trainer.training()
