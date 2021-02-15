from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

MetaData = namedtuple("MetaData", ["init_val", "start_time", "end_time", "time_step", "loss_type"])


def set_seeds_and_clear_session():
    keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)
