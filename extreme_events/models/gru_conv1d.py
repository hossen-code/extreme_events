import tensorflow as tf
from tensorflow import keras


class GRUAndConv1D(object):
    """Implements GRU and Conv1D as the first layer model for extreme event prediction"""
    def __init__(self,
                 loss_function,
                 optimizer,
                 metrics,
                 input_shape,
                 num_gru_layers=2,
                 num_gru_nodes_per_layer=20,
                 ):
        self.num_gru_layers = num_gru_layers
        self.num_gru_nodes_per_layer = num_gru_nodes_per_layer
        self.input_shape = input_shape
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.metrics = metrics
        self.model = None
        self.hisory = None

    def fit(self, X, y, num_epochs=20, on_gpu=True):
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
        conv_1d_layer = keras.layers.Conv1D(filters=20, # these numbers just from the hands on ml book
                                            kernel_size=2,
                                            strides=4,
                                            padding="valid",
                                            input_shape=self.input_shape)
        layers.append(conv_1d_layer)
        # TODO, conv1d works with return sequence false, and last layer not a time distributed
        layers.append(keras.layers.GRU(self.num_gru_nodes_per_layer))
        layers.append(keras.layers.TimeDistributed(keras.layers.Dense(1)))
        model = keras.models.Sequential(layers)
        model.compile(loss=self.loss_function,
                      optimizer=self.optimizer,
                      metrics=self.metrics)

        model.compile(loss="mse", optimizer="adam", metrics=["mse"])

        self.model = model
