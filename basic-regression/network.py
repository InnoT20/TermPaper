import keras
import tensorflow as tf
from keras import layers
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import LambdaCallback
from numpy import linalg as LA


class Model:

    def __init__(self, input_units, hidden_units, epochs):
        self.epochs = epochs

        model = keras.Sequential([
            layers.Dense(64, activation=tf.nn.relu, input_shape=[input_units]),
            layers.Dense(64, activation=tf.nn.relu),
            layers.Dense(1)
        ])

        model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse'])

        self.model = model
        self.history = None

        self.history_weights = []

        # Параметр patience определяет количество эпох, которые можно пропустить без улучшений
        self.early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
        self.print_weights = LambdaCallback(
            on_batch_end=lambda batch, logs: self.history_weights.append(LA.norm(model.layers[0].get_weights()[0][0])))

    def fit(self, train_data, train_labels):
        self.history = self.model.fit(train_data, train_labels, epochs=self.epochs, validation_split=0.2, verbose=1,
                                      batch_size=20, callbacks=[self.print_weights, self.early_stop])
        return self.history

    def save_weight(self, filename="mnist_model.h5"):
        self.model.save_weights(filename)

    def evaluate(self, test_data, test_label):
        return self.model.evaluate(test_data, test_label, verbose=0)

    def plot_history(self):
        _hist = pd.DataFrame(self.history.history)
        _hist['epoch'] = self.history.epoch

        plt.figure(figsize=(8, 12))

        plt.subplot(2, 1, 1)
        plt.xlabel('Эпоха')
        plt.ylabel('Среднее абсолютное отклонение')
        plt.plot(_hist['epoch'], _hist['mean_absolute_error'], label='Ошибка при обучении')
        plt.plot(_hist['epoch'], _hist['val_mean_absolute_error'], label='Ошибка при проверке')
        plt.ylim([0, 5])
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.xlabel('Эпоха')
        plt.ylabel('Среднеквадратическая ошибка')
        plt.plot(_hist['epoch'], _hist['mean_squared_error'], label='Ошибка при обучении')
        plt.plot(_hist['epoch'], _hist['val_mean_squared_error'], label='Ошибка при проверке')
        plt.ylim([0, 20])
        plt.legend()
        plt.show()
        # plt.savefig('foo.png')
