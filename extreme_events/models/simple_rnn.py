# trying to run a simple rnn model first

from tensorflow import keras
import numpy as np

def generate_time_series(batch_size, n_steps):
    # this is from the hands-on ml book
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10)) # wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5) # + noise
    return series[..., np.newaxis].astype(np.float32)


if __name__ == "__main__":
    n_steps = 50
    series = generate_time_series(1, n_steps + 10)
    X_new, Y_new = series[:, :n_steps], series[:, n_steps:]
    X = X_new

    model = keras.models.Sequential([
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.SimpleRNN(20),
        keras.layers.Dense(1)
    ])

    for step_ahead in range(10):
        y_pred_one = model.predict(X[:, step_ahead:])[:, np.newaxis, :]
    X = np.concatenate([X, y_pred_one], axis=1)
    Y_pred = X[:, n_steps:]
