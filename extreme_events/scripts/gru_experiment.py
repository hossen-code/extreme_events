from tensorflow import keras

from extreme_events.data_source.data_providers import rossler_dataset_maker
from extreme_events.utils.model_utils import set_seeds_and_clear_session
from extreme_events.models.gru import GRUAndConv1D


if __name__ == "__main__":
    set_seeds_and_clear_session()
    x_0 = [2, 2, 2]
    start_time = 0.0
    end_time = 30.0
    time_step = 0.01
    data = rossler_dataset_maker(x_0=x_0, start_time=start_time, end_time=end_time, time_step=time_step)
    x_train = data[0][:, :-1].reshape(1, data.shape[1], 2)  # both x and y
    y_train = data[0][:, -1:].reshape(1, data.shape[1], 1)  # assuming the last column is target
    optimizer = keras.optimizers.Adam(lr=0.005)
    loss = keras.losses.Huber()
    metrics = ["mse"]
    model = GRUAndConv1D(loss_function=loss, optimizer=optimizer, metrics=metrics)
    model.fit(X=x_train, y=y_train)
