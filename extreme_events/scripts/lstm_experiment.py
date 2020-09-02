from tensorflow import keras
import numpy as np

from extreme_events.data_source.data_providers import rossler_dataset_maker
from extreme_events.utils.model_utils import set_seeds_and_clear_session
from extreme_events.models.lstm import LSTM


if __name__ == "__main__":
    set_seeds_and_clear_session()
    x_0 = [2, 2, 2]
    start_time = 0.0
    end_time = 30.0
    time_step = 0.01
    data = rossler_dataset_maker(x_0=x_0, start_time=start_time, end_time=end_time, time_step=time_step)
    x_train = data[0][:, :-1].reshape(1, 3000, 2) # both x and y (assuming 2 variates)
    y_train = data[0][:, -1:].reshape(1, 3000, 1) # assuming the last column is target
    optimizer = keras.optimizers.Adam(lr=0.005)

    def last_time_step_mse(Y_true, Y_pred):
        return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

    loss = "mse"
    metrics = ["mse"]
    model = LSTM(loss_function=loss, optimizer=optimizer, metrics=metrics)
    model.fit(X=x_train, y=y_train)
