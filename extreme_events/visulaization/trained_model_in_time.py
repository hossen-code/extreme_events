import pickle
from pathlib import Path

from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow_addons.losses import SigmoidFocalCrossEntropy

from extreme_events.data_source.data_providers import rossler_dataset_maker, if_flips_in_next_n_steps

MODEL_NAME = "simple_rnn_cross_entropy_loss"
MODEL_PATH = Path.home() / f"Desktop/Repos/extreme_events/saved_models/{MODEL_NAME}"


def load_model_and_viz():
    loaded_model = keras.models.load_model(
        f'/home/hossein/Desktop/Repos/extreme_events/saved_models/{MODEL_NAME}')
    with open(MODEL_PATH / "meta_data", "rb") as meta_data_file:
        metadata = pickle.load(meta_data_file)

    test_time = 10.0
    data = rossler_dataset_maker(x_0=metadata.init_val,
                                 start_time=metadata.start_time,
                                 end_time=metadata.end_time + test_time, # extra 10 time unites for test
                                 time_step=metadata.time_step)

    train_time_steps = int((metadata.end_time - metadata.start_time) / metadata.time_step)
    test_time_steps = int(test_time / metadata.time_step)
    tot_time_steps = train_time_steps + test_time_steps

    all_x = data[:, :, :-1].reshape(1, tot_time_steps, 2)
    all_y = data[:, :, -1].reshape(1, tot_time_steps)
    all_y_if_flipped = if_flips_in_next_n_steps(all_y, threshold=5.0, n_time_steps=20)

    plt.plot(data[0][0:tot_time_steps, 2:3].T.tolist()[0])

    plt.plot([None] * train_time_steps + all_y_if_flipped[0][train_time_steps:tot_time_steps].tolist())

    results2 = []
    for i in range(test_time_steps):
        val = loaded_model.predict(all_x[:, i:train_time_steps + i, :])
        if val[0][0] > 0:
            results2.append(1)
        else:
            results2.append(0)

        if i % 10 == 0:
            print(f"done {i} iteration")

    plt.plot([None] * train_time_steps + results2, "r")
    plt.show()


if __name__ == "__main__":
    load_model_and_viz()


