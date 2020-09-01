import tensorflow as tf
from tensorflow import keras


class LSTM(object):
    """Implements LSTM model for extreme event prediction"""
    def __init__(self,
                 loss_function,
                 optimizer,
                 metrics,
                 num_lstm_layers=2,
                 num_lstm_nodes_per_layer=20,
                 num_variates=2,
                 ):
        self.num_lstm_layers = num_lstm_layers
        self.num_lstm_nodes_per_layer = num_lstm_nodes_per_layer
        self.num_variates = num_variates
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.metrics = metrics
        self.model = None
        self.hisory = None

    def fit(self, X, y, num_epochs=20, on_gpu=True):
        # memory growth has to be added for lstm to run
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

        if not self.model:
            self._make_model()
        if on_gpu:
            with tf.device("/GPU:0"):
                self.history = self.model.fit(X, y, epochs=num_epochs)
        else:
            self.history = self.model.fit(X, y, epochs=num_epochs)

        return self.history

    def _make_model(self):
        # model = keras.Sequential()
        layers = []
        # adding input layer
        input_layer = keras.layers.LSTM(self.num_lstm_nodes_per_layer,
                                        return_sequences=True,
                                        input_shape=[None, self.num_variates])
        layers.append(input_layer)
        # adding hidden layers
        for _ in range(self.num_lstm_layers - 1):
            layers.append(keras.layers.LSTM(self.num_lstm_nodes_per_layer,
                                        return_sequences=True))
        # adding output layer
        output_layer = keras.layers.TimeDistributed(keras.layers.Dense(10)) # TODO: this number is arbitrary ATM
        layers.append(output_layer)
        model = keras.models.Sequential(layers)
        model.compile(loss=self.loss_function,
                      optimizer=self.optimizer,
                      metrics=self.metrics)
        self.model = model

