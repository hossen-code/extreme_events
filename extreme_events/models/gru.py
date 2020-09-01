import tensorflow as tf
from tensorflow import keras


class GRUAndConv1D(object):
    """Implements GRU and Conv1D as the first layer model for extreme event prediction"""
    def __init__(self,
                 loss_function,
                 optimizer,
                 metrics,
                 num_gru_layers=2,
                 num_gru_nodes_per_layer=20,
                 num_variates=2,
                 ):
        self.num_gru_layers = num_gru_layers
        self.num_gru_nodes_per_layer = num_gru_nodes_per_layer
        self.num_variates = num_variates
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.metrics = metrics
        self.model = None
        self.hisory = None

    def fit(self, X, y, num_epochs=20, on_gpu=True):
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
        conv_1d_layer = keras.layers.Conv1D(filters=20, # these numbers just from the hands on ml book
                                            kernel_size=4,
                                            strides=2,
                                            padding="valid",
                                            input_shape=[None, self.num_variates])
        layers.append(conv_1d_layer)
        for _ in range(self.num_gru_layers):
            layers.append(keras.layers.GRU(self.num_gru_nodes_per_layer,
                                           return_sequences=True))

        layers.append(keras.layers.TimeDistributed(keras.layers.Dense(10)))
        model = keras.models.Sequential(layers)
        model.compile(loss=self.loss_function,
                      optimizer=self.optimizer,
                      metrics=self.metrics)
        self.model = model
