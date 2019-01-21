# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from SNU_layer import SNU
from chainer import Variable

""" Build Spiking Neural Unit """
num_time = 100 # simulation time step
V_th = 2.5
tau = 25
dt = 1

snu_l = SNU(n_in=1, n_out=1, l_tau=(1-dt/tau), soft=False)
snu_l.b_th.b = Variable(np.array([-V_th], dtype=np.float32))
snu_l.Wx.W = Variable(np.array([[1.0]], dtype=np.float32))


""" Generate Poisson Spike Trains """
frequency = 10
num_spikes_per_cell = 20
x = np.zeros(num_time) # input spike array
isi = np.random.poisson(frequency, num_spikes_per_cell)
idx = np.cumsum(isi)
idx = idx[idx<num_time]
x[idx] = 1
x = np.array([[x]]).astype(np.float32)

""" Simulation """
s_arr = np.zeros(num_time) # array to save membrane potential
y_arr = np.zeros(num_time) # array to save output

for i in range(num_time):    
    y = snu_l(x[:, :, i])
    
    s_arr[i] = snu_l.s.array
    y_arr[i] = y.array

""" Plot results """    
plt.figure(figsize=(6,6))

plt.subplot(3,1,1)
plt.title("Spiking Neural Unit")
plt.plot(x[0,0])
plt.ylabel("Input")

plt.subplot(3,1,2)
plt.plot(s_arr)
plt.ylabel('Membrane\n Potential')

plt.subplot(3,1,3)
plt.plot(y_arr)
plt.ylabel("Output")
plt.xlabel("Time (ms)")

plt.tight_layout()
plt.savefig("SNU_result.png")