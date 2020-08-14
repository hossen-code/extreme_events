import tensorflow as tf
from tensorflow import keras
import numpy as np

from extreme_events.data_source.data_providers import train_test_splitter, \
    rossler_dataset_maker


def set_seeds_and_clear_session():
    keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)


def make_lstm_model():
    model = keras.models.Sequential(
        [keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
         # keras.layers.LSTM(20, return_sequences=True),
         # keras.layers.TimeDistributed(keras.layers.Dense(10))
         ])
    optimizer = keras.optimizers.SGD(lr=1e-8, momentum=0.9)
    model.compile(loss=keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mse"])
    return model


def simple_rnn_model():
    model = keras.models.Sequential([
        keras.layers.SimpleRNN(1, input_shape=[None, 1])])
    optimizer = keras.optimizers.SGD(lr=1e-8, momentum=0.9)
    model.compile(loss=keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mse"])
    return model


def run_model(train_data):
    # lstm = make_lstm_model()
    lstm = simple_rnn_model()
    lr_schedule = keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-8 * 10**(epoch / 20))
    history = lstm.fit(train_data, epochs=10, callbacks=[lr_schedule])

    return history


# def split_data_stream(data_tuple, train_percentage):
#     for column in data_tuple.feature:
#         train, test = train_test_splitter(column, train_percentage)
#     return train, test




if __name__ == "__main__":
    set_seeds_and_clear_session()
    x_0 = [2, 2, 2]
    time_range = np.arange(0, 30, 0.1)
    data = rossler_dataset_maker(x_0=x_0, time_range=time_range)
    train, test = train_test_splitter(data.feature[0], train_ratio=0.8)
    run_model(train)
