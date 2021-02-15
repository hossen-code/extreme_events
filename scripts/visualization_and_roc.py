from tensorflow import keras
from tensorflow_addons.losses import SigmoidFocalCrossEntropy

from extreme_events.data_source.data_providers import rossler_dataset_maker, if_flips_in_next_n_steps

MODEL_NAME = ""
loaded_model = keras.models.load_model(
    '/home/hossein/Desktop/Repos/extreme_events/saved_models/simple_rnn_cross_entropy_loss')

from tensorflow.keras.utils import get_custom_objects
# loss = SigmoidFocalCrossEntropy()
# get_custom_objects().update({"loss": loss})

x_0 = [2, 2, 2]
start_time = 0.0
end_time = 40.0
time_step = 0.01
data = rossler_dataset_maker(x_0=x_0,
                             start_time=start_time,
                             end_time=end_time,
                             time_step=time_step)
x_train = data[:, 0:3000, :-1].reshape(1, 3000, 2)  # both x and y
y_train = data[:, 0:3000, -1:].reshape(1, 3000)

x_test = data[:, 3000:4000, :-1].reshape(1, 1000, 2)
y_test = data[:, 3000:4000, -1:].reshape(1, 1000)


all_x = data[:, :, :-1].reshape(1, 4000, 2)
all_y = data[:, :, -1].reshape(1, 4000)
all_y = if_flips_in_next_n_steps(all_y,  threshold=5.0, n_time_steps=20)

from matplotlib import pyplot as plt
import numpy as np

# from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure(figsize=(13, 9))
# ax = fig.gca(projection='3d')
# ax.set_ylim(-6, 6)
# ax.set_xlim(-6, 6)
# ax.set_zlim(0, 12)
# ax.view_init(20, 160)
# ax.set_axis_off()
# ax.plot(data[0][0:3000, 0:1].T.tolist()[0],
#         data[0][0:3000, 1:2].T.tolist()[0],
#         data[0][0:3000, 2:3].T.tolist()[0], 'magenta')
# plt.show()

plt.plot(data[0][0:4000, 2:3].T.tolist()[0])

plt.plot([None]*3000 + all_y[0][3000:4000])

# results = []
# for i in range(1000):
#     val = loaded_model.predict(np.array([[x_test[0][i]]]))
#     results.append(val[0][0])
# plt.plot([None]*3000 + results)

results2 = []
for i in range(1000):
    val = loaded_model.predict(all_x[:, i:3000+i, :])
    if val[0][0] > 0:
        results2.append(1)
    else:
        results2.append(0)

    if i % 10 == 0:
        print(f"done {i} iteration")


plt.plot([None]*3000 + results2, "r")
plt.show()
pass