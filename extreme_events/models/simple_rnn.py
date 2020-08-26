# trying to run a simple rnn model first
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def generate_time_series(batch_size, n_steps):
    # this is from the hands-on ml book
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10)) # wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5) # + noise
    return series[..., np.newaxis].astype(np.float32)


def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, 20, 0, 0.05])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)


if __name__ == "__main__":
    n_steps = 50
    series = generate_time_series(10000, n_steps + 1)
    X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
    X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
    X_test, y_test = series[9000:, :n_steps], series[9000:, -1]
    print(X_train.shape, y_train.shape)
    model = keras.models.Sequential([
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.SimpleRNN(20),
        keras.layers.Dense(1)
    ])

    # model = keras.models.Sequential([
    #     keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
    #     keras.layers.LSTM(20, return_sequences=True),
    #     keras.layers.TimeDistributed(keras.layers.Dense(10))
    # ])


    def last_time_step_mse(Y_true, Y_pred):
        return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

    optimizer = keras.optimizers.Adam(lr=0.01)
    model.compile(loss="mse", optimizer=optimizer, metrics=[last_time_step_mse])
    with tf.device('/GPU:0'):
        history = model.fit(X_train, y_train, epochs=20,
                            validation_data=(X_valid, y_valid))

    model.evaluate(X_valid, y_valid)
    plot_learning_curves(history.history["loss"], history.history["val_loss"])
    plt.show()
    plt.savefig()

    # for step_ahead in range(10):
    #     y_pred_one = model.predict(X_train[:, step_ahead:])[:, np.newaxis, :]
    # X = np.concatenate([X, y_pred_one], axis=1)
    # Y_pred = X[:, n_steps:]
    #
    # print(Y_pred)
