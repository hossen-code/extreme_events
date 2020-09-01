from tensorflow import keras
import numpy as np

from extreme_events.data_source.data_providers import rossler_dataset_maker
from extreme_events.utils.model_utils import set_seeds_and_clear_session
from extreme_events.models.simple_rnn import SimpleRNN

if __name__ == "__main__":
    set_seeds_and_clear_session()
    x_0 = [2, 2, 2]
    time_range = np.arange(0, 30, 0.01)
    data = rossler_dataset_maker(x_0=x_0, time_range=time_range)
    x_train = data[0][:, :-1].reshape(1, 3000, 2) # both x and y
    y_train = data[0][:, -1:].reshape(1, 3000, 1) # assuming the last column is target
    optimizer = keras.optimizers.Adam(lr=0.005)
    loss = keras.losses.Huber()
    metrics = ["mse"]
    model = SimpleRNN(loss_function=loss, optimizer=optimizer, metrics=metrics)
    model.fit(X=x_train, y=y_train)