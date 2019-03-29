from keras.datasets import mnist

from predictor_lstm import LSTMPredictor
from predictor_mlp import MLPPredictor

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    predictor_mlp = MLPPredictor(x_train, y_train, x_test, y_test)
    predictor_mlp.fit()
    predictor_mlp.evaluate()

    predictor_lstm = LSTMPredictor(x_train, y_train, x_test, y_test)
    predictor_lstm.fit()
    predictor_lstm.evaluate()
