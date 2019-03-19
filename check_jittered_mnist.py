# -*- coding: utf-8 -*-

import os
import chainer
import numpy as np
import matplotlib.pyplot as plt

img_save_dir = "./imgs/"
os.makedirs(img_save_dir, exist_ok=True)

# Load the MNIST dataset
train, _ = chainer.datasets.get_mnist()

i = 2
y = train[i][0]
y = np.heaviside(y, 0)
label = train[i][1]

num_time = 20
fr = 100 # Hz
dt = 1e-3 # sec

y_fr = fr * np.repeat(np.expand_dims(y, 1), num_time, axis=1)

x = np.where(np.random.rand(784, num_time) < y_fr*dt, 1, 0)
sum_x = np.sum(x, axis=1)

fig = plt.figure(figsize=(6, 3))
ax1 = fig.add_subplot(1, 3, 1)
ax1.set_title("Binarized")
ax1.imshow(np.reshape(y, (28, 28)))

ax2 = fig.add_subplot(1, 3, 2)
ax2.set_title("Jittered\n (one time step)")
ax2.imshow(np.reshape(x[:, 1], (28, 28)))

ax3 = fig.add_subplot(1, 3, 3)
ax3.set_title("Jittered\n (sum all time step)")
ax3.imshow(np.reshape(sum_x, (28, 28)))
#plt.show()
plt.savefig(img_save_dir+"JitteredMNIST")