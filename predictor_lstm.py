"""Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
"""
from keras import Sequential
from keras.layers import Dense, LSTM, Activation
from keras.optimizers import RMSprop
from keras.utils import to_categorical

from predictor import Predictor


class LSTMPredictor(Predictor):
    batch_size = 128
    epochs = 20

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train.reshape(-1, 28, 28).astype('float32') / 255
        self.x_test = x_test.reshape(-1, 28, 28).astype('float32') / 255

        self.y_train = to_categorical(y_train, self.num_classes)
        self.y_test = to_categorical(y_test, self.num_classes)

        self.model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(LSTM(128, batch_input_shape=(None, 28, 28), unroll=True))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])

        return model

    def fit(self):
        self.model.fit(self.x_train, self.y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=1,
                       validation_data=(self.x_test, self.y_test))

    def evaluate(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])


