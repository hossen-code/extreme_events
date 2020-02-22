
import tensorflow as tf
import numpy as np

from extreme_events.data_source.data_providers import train_test_splitter


def set_seeds_and_clear_session():
    tf.keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)


def make_lstm_model():
    model = tf.keras.models.Sequential([tf.keras.layers.Lambda(
        lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 100.0) # not sure why multiplied by 100
    ])
    optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])
    return model


def run_model(train_data):
    lstm = make_lstm_model()
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-8 * 10**(epoch / 20))
    history = lstm.fit(train_data, epochs=100, callbacks=[lr_schedule])

    return history


def split_data_stream(data_tuple, train_percentage):
    for column in data_tuple.feature:
        train, test = train_test_splitter(column, train_percentage)
