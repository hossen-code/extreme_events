from tensorflow import keras
import numpy as np

from extreme_events.data_source.data_providers import rossler_dataset_maker
from extreme_events.utils.model_utils import set_seeds_and_clear_session
from extreme_events.models.simple_rnn import SimpleTwoLayerRNN


if __name__ == "__main__":
    set_seeds_and_clear_session()
    x_0 = [2, 2, 2]
    start_time = 0.0
    end_time = 30.0
    time_step = 0.01
    data = rossler_dataset_maker(x_0=x_0,
                                 start_time=start_time,
                                 end_time=end_time,
                                 time_step=time_step)
    x_train = data[0][:, :-1].reshape(1, data.shape[1], 2)  # both x and y
    y_train = data[0][:, -1:].reshape(1, data.shape[1])  # assuming the last column is target
    optimizer = keras.optimizers.Adam(lr=0.005)
    loss = keras.losses.Huber()
    metrics = ["mse"]
    model = SimpleTwoLayerRNN(loss_function=loss, optimizer=optimizer, metrics=metrics,
                              input_shape=(x_train.shape[1], x_train.shape[2]))
    model.fit(X=x_train, y=y_train)
