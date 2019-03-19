# -*- coding: utf-8 -*-
""" Analysis. """

import os
import argparse

import chainer
from chainer import serializers
from chainer import cuda
#import chainer.functions as F
xp = cuda.cupy

import network

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_data(N=1, dt=5e-3, num_time=20, max_fr=60):
    """ geneate input data. """
    _, test = chainer.datasets.get_mnist()
    x = np.zeros((N, 784, num_time)) # 784=28x28
    y = np.zeros(N)
    for i in tqdm(range(N)):    
        fr = max_fr * np.repeat(np.expand_dims(np.heaviside(test[i][0],0), 1), num_time, axis=1)
        x[i] = np.where(np.random.rand(784, num_time) < fr*dt, 1, 0)
        y[i] = test[i][1]
    
    return x.astype(np.float32), y.astype(np.int8)

def plot_activation(model, N, dt, num_time, n_mid,
                    n_out, max_fr, gpu):
    x, y = load_data(N=N, dt=dt, num_time=num_time, max_fr=max_fr)
    
    idx = 0
    check_x = np.sum(x[idx], axis=1)
    
    plt.figure(figsize=(3,3))
    plt.imshow(np.reshape(check_x, (28, 28)))
    plt.savefig("input_all_sum.png")
    #plt.show()
    
    plt.figure(figsize=(3,3))
    plt.imshow(np.reshape(x[idx, :, 1], (28, 28)))
    plt.savefig("input_frame.png")
    #plt.show()
    
    if gpu >= 0:
        cuda.get_device_from_id(0).use()
        model.to_gpu()
        x = cuda.cupy.array(x)
        y = cuda.cupy.array(y)
       
    with chainer.using_config('train', False):
        loss, accuracy, h1_list, h2_list, out_list = model(x, y)
    
    h1_all = np.zeros((num_time, N, n_mid))
    h2_all = np.zeros((num_time, N, n_mid))
    out_all = np.zeros((num_time, N, n_out))
    for i in tqdm(range(num_time), desc="Getting activation"):
        h1_all[i] = cuda.to_cpu(h1_list[i].data)
        h2_all[i] = cuda.to_cpu(h2_list[i].data)
        out_all[i] = cuda.to_cpu(out_list[i].data)
    
    t = np.arange(0, num_time)*dt*1000
    
    plt.figure(figsize=(4,4))
    plt.ylim(-0.5, n_mid-0.5)
    plt.ylabel("# Unit")
    plt.xlabel("Simulation Time(ms)")
    for i in range(n_mid):
        spk = np.where(h1_all[:, idx, i]==1, i, -1)
        plt.scatter(t, spk, color="r",
                    s=0.1)
    
    plt.savefig("h1.png")
    
    plt.figure(figsize=(4,4))
    plt.ylim(-0.5, n_mid-0.5)
    plt.ylabel("# Unit")
    plt.xlabel("Simulation Time(ms)")
    for i in range(n_mid):
        spk = np.where(h2_all[:, idx, i]==1, i, -1)
        plt.scatter(t, spk, color="r",
                    s=0.1)
    
    plt.savefig("h2.png")
    
    
    plt.figure(figsize=(4,4))
    plt.ylim(-0.5, n_out-0.5)
    plt.ylabel("# Unit")
    plt.xlabel("Simulation Time(ms)")
    plt.yticks(np.arange(0, n_out).tolist())
    for i in range(n_out):
        spk = np.where(out_all[:, idx, i]==1, i, -1)
        plt.scatter(t, spk, color="r", marker="|")
    
    plt.savefig("result.png")
    
    #plt.show()
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU number to training.')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Load saved model (model filename).')
    parser.add_argument('--batch', '-b', type=int, default=32,
                        help='Mini batch size.')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Total training epoch.')
    parser.add_argument('--dt', '-dt', type=int, default=1e-3,
                        help='Simulation time step size (sec).')
    parser.add_argument('--ndata', '-nd', type=int, default=1,
                        help='The number of analysis trials.')
    parser.add_argument('--freq', '-f', type=float, default=100,
                        help='Input signal maximum frequency (Hz).')
    parser.add_argument('--time', '-t', type=int, default=5,
                        help='Total simulation time steps.')
    args = parser.parse_args()
    
    img_save_dir = "./imgs/"
    os.makedirs(img_save_dir, exist_ok=True)
    
    n_mid = 256
    n_out = 10
    
    chainer.global_config.autotune = True
    if args.gpu >= 0:
        # Make a specified GPU current
        model = network.SNU_Network(n_in=784, n_mid=n_mid, n_out=n_out,
                                    num_time=args.time, gpu=True,
                                    test_mode=True)

        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU
    else:
        model = network.SNU_Network(n_in=784, n_mid=n_mid, n_out=n_out,
                                    num_time=args.time, gpu=False,
                                    test_mode=True)

    if args.model != None:
        print( "Loading model from " + args.model)
        serializers.load_npz(args.model, model)
    
      
    plot_activation(model=model, N=args.ndata, dt=args.dt,
                    num_time=args.time, n_mid=n_mid,
                    n_out=n_out, 
                    max_fr=args.freq, gpu=args.gpu)
    
            
if __name__ == '__main__':
    main()
