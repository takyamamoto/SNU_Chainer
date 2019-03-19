# -*- coding: utf-8 -*-

import argparse

import chainer
from chainer import training
from chainer.training import extensions
from chainer import iterators, optimizers, serializers

import numpy as np
from chainer import dataset
from tqdm import tqdm

from chainer import cuda
xp = cuda.cupy


#import matplotlib.pyplot as plt

from model import network

np.random.seed(seed=0)

class LoadDataset(dataset.DatasetMixin):
    def __init__(self, N=60000, dt=1e-3, num_time=100, max_fr=60):
        train, _ = chainer.datasets.get_mnist()
        x = np.zeros((N, 784, num_time)) # 784=28x28
        y = np.zeros(N)
        for i in tqdm(range(N)):    
            fr = max_fr * np.repeat(np.expand_dims(np.heaviside(train[i][0],0), 1), num_time, axis=1)
            x[i] = np.where(np.random.rand(784, num_time) < fr*dt, 1, 0)
            y[i] = train[i][1]
        
        self.x = x.astype(np.float32)
        self.y = y.astype(np.int8)
        self.N = N
        
    def __len__(self):
        return self.N

    def get_example(self, i):
        return self.x[i], self.y[i]
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--model', '-m', type=str, default=None)
    parser.add_argument('--batch', '-b', type=int, default=128)
    parser.add_argument('--epoch', '-e', type=int, default=100)
    parser.add_argument('--ndata', '-nd', type=int, default=60000,
                        help='The number of analysis trials (<=60000).')
    parser.add_argument('--time', '-t', type=int, default=10,
                        help='Total simulation time steps.')
    parser.add_argument('--dt', '-dt', type=float, default=1e-3,
                        help='Simulation time step size (sec).')
    parser.add_argument('--freq', '-f', type=float, default=100,
                        help='Input signal maximum frequency (Hz).')
    parser.add_argument('--lr', '-l', type=float, default=1e-4)
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    args = parser.parse_args()

    print("Loading datas")
    dataset = LoadDataset(N=args.ndata, dt=args.dt, 
                          num_time=args.time, max_fr=args.freq)
    
    #plt.imshow(np.reshape(dataset[1][0][:, 0], (28, 28)))
    #plt.show()
    
    val_rate = 0.2
    split_at = int(len(dataset) * (1-val_rate))
    train, val = chainer.datasets.split_dataset(dataset, split_at)
    
    train_iter = iterators.SerialIterator(train, batch_size=args.batch, shuffle=True)
    test_iter = iterators.SerialIterator(val, batch_size=args.batch, repeat=False, shuffle=False)
    
    chainer.global_config.autotune = True
    
    # Set up a neural network to train.
    print("Building model")

    if args.gpu >= 0:
        # Make a specified GPU current
        model = network.SNU_Network(n_in=784, n_mid=256, n_out=10,
                                    num_time=args.time, gpu=True)

        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU
    else:
        model = network.SNU_Network(n_in=784, n_mid=256, n_out=10,
                                    num_time=args.time, gpu=False)

    
    optimizer = optimizers.Adam(alpha=args.lr)
    #optimizer = optimizers.SGD(lr=args.lr)
    #optimizer = optimizers.RMSprop(lr=args.lr, alpha=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(1e-4))
    
    if args.model != None:
        print( "loading model from " + args.model)
        serializers.load_npz(args.model, model)
    
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='results')
    
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.LogReport(trigger=(100, 'iteration')))
    
    # Snapshot
    trainer.extend(extensions.snapshot_object(model, 'model_snapshot_{.updater.epoch}'),
                                              trigger=(1,'epoch'))
    
    trainer.extend(extensions.ExponentialShift('alpha', 0.5),trigger=(5, 'epoch'))

    # Save two plot images to the result dir
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'), trigger=(1, 'epoch'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'), trigger=(1, 'epoch'))
    
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']), trigger=(1, 'iteration'))
    
    trainer.extend(extensions.ProgressBar(update_interval=1))
    
    # Train
    trainer.run()
    
    # Save results
    print("Optimization Finished!")
    modelname = "./results/model"
    print( "Saving model to " + modelname)
    serializers.save_npz(modelname, model)

if __name__ == '__main__':
    main()
