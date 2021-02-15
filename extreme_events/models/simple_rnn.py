import tensorflow as tf
from tensorflow import keras


class SimpleTwoLayerRNN(object):
    """Implements simple RNN model for extreme event prediction"""
    def __init__(self,
                 loss_function,
                 optimizer,
                 metrics,
                 input_shape,
                 num_rnn_nodes_per_layer=20,
                 ):
        self.num_rnn_nodes_per_layer = num_rnn_nodes_per_layer
        self.input_shape = input_shape
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.metrics = metrics
        self.model = None
        self.hisory = None

    def fit(self, X, y, num_epochs=10, on_gpu=True):
        # memory growth has to be set to True for lstm to run
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
        layers = []
        input_layer = keras.layers.SimpleRNN(self.num_rnn_nodes_per_layer,
                                             return_sequences=True,
                                             input_shape=self.input_shape)
        layers.append(input_layer)
        # adding hidden layers
        layers.append(keras.layers.SimpleRNN(self.num_rnn_nodes_per_layer,
                                             return_sequences=True))
        layers.append(keras.layers.SimpleRNN(self.num_rnn_nodes_per_layer,
                                             return_sequences=True))
        # adding output layer
        layers.append(keras.layers.Dense(1))
        model = keras.models.Sequential(layers)
        model.compile(loss=self.loss_function,
                      optimizer=self.optimizer,
                      metrics=self.metrics)
        self.model = model
