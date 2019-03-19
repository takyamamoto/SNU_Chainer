# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

from model import snu_layer
from chainer import Variable

img_save_dir = "./imgs/"
os.makedirs(img_save_dir, exist_ok=True)
    

""" Build Spiking Neural Unit """
num_time = 100 # simulation time step
V_th = 2.5
tau = 25e-3 # sec
dt = 1e-3 # sec

snu_l = snu_layer.SNU(n_in=1, n_out=1, l_tau=(1-dt/tau),
                      soft=False, initial_bias=-V_th)
snu_l.Wx.W = Variable(np.array([[1.0]], dtype=np.float32))


""" Generate Poisson Spike Trains """
fr = 100 # Hz
x = np.where(np.random.rand(1, num_time) < fr*dt, 1, 0)
x = np.expand_dims(x, 0).astype(np.float32)

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
plt.savefig(img_save_dir+"Check_SNU_result.png")