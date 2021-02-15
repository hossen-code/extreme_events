import pickle
from pathlib import Path

from tensorflow import keras
from matplotlib import pyplot
from tensorflow.python.keras.utils import losses_utils
from tensorflow_addons.losses import SigmoidFocalCrossEntropy

from extreme_events.data_source.data_providers import rossler_dataset_maker, if_flips_in_next_n_steps
from extreme_events.utils.model_utils import set_seeds_and_clear_session, MetaData
from extreme_events.models.simple_rnn import SimpleTwoLayerRNN

if __name__ == "__main__":
    set_seeds_and_clear_session()
    x_0 = [2, 2, 2]
    start_time = 0.0
    end_time = 30.0
    time_step = 0.01
    loss_type = 'binary_crossentropy'
    model_meta_data = MetaData(init_val=x_0, start_time=start_time, end_time=end_time, time_step=time_step, loss_type=loss_type)

    data = rossler_dataset_maker(x_0=model_meta_data.init_val,
                                 start_time=model_meta_data.start_time,
                                 end_time=model_meta_data.end_time,
                                 time_step=model_meta_data.time_step)
    x_train = data[0][:, :-1].reshape(1, data.shape[1], 2)  # both x and y
    y_train = data[0][:, -1:].reshape(1, data.shape[1])  # assuming the last column is target
    y_train = if_flips_in_next_n_steps(y_train, threshold=5.0, n_time_steps=20)
    optimizer = keras.optimizers.Adam(lr=0.005)
    # loss = keras.losses.Huber()
    loss = keras.losses.BinaryCrossentropy(
        from_logits=False, label_smoothing=0, reduction=keras.losses.Reduction.SUM,
    )

    # loss = SigmoidFocalCrossEntropy()
    metrics = ["mse"]
    model = SimpleTwoLayerRNN(loss_function=loss, optimizer=optimizer, metrics=metrics,
                              input_shape=(x_train.shape[1], x_train.shape[2]))
    history = model.fit(X=x_train, y=y_train)

    # plot history
    pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    model_name = "simple_rnn_cross_entropy_loss"
    saving_path = Path.home() / f"Desktop/Repos/extreme_events/saved_models/{model_name}"
    pyplot.savefig(f"{saving_path}/{model_name}_training.png")
    model.model.save(saving_path)
    filename = 'meta_data'
    with open(saving_path / filename, 'wb') as opened_file:
        pickle.dump(model_meta_data, opened_file)


    pass
