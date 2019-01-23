# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L

import numpy as np
from chainer import cuda

from chainer import Parameter
from chainer import initializers

from chainer import Variable
from step_func import step

# Building Spiking Neural Unit
class SNU(chainer.Chain):
    """
    Args:
        n_in (int): The number of input.
        n_out (int): The number of output.
        l_tau (floot): Degree of leak (From 0 to 1).
        soft (bool): Change output activation to sigmoid func (True)
                     or Step func. (False)
        rec (bool): Adding recurrent connection or not.
    """
    def __init__(self, n_in, n_out, l_tau=0.8, soft=False,
                 rec=False, nobias=False, initial_bias=None,
                 gpu=False):
        super(SNU, self).__init__()
        with self.init_scope():
            self.Wx = L.Linear(n_in, n_out, nobias=True)
            if rec:
                self.Wy = L.Linear(n_out, n_out, nobias=True)
            
            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    bias_initializer = initializers.Uniform()
                    self.b = Parameter(bias_initializer, n_out)
                else:                    
                    bias_initializer = initializers._get_initializer(initial_bias)
                    self.b = Parameter(bias_initializer, n_out)
            
            self.n_out = n_out            
            self.l_tau = l_tau
            self.rec = rec
            self.soft = soft
            
            self.gpu = gpu
            
            self.s = None
            self.y = None

    def reset_state(self, s=None, y=None):
        self.s = s
        self.y = y

    def initialize_state(self, shape):
        if self.gpu:
            xp = cuda.cupy
        else:                
            xp = np
        self.s = Variable(xp.zeros((shape[0], self.n_out), dtype=xp.float32))
        self.y = Variable(xp.zeros((shape[0], self.n_out), dtype=xp.float32))
        
    def __call__(self, x):
        if self.s is None:
            self.initialize_state(x.shape)
        
        if self.rec:            
            s = F.relu(self.Wx(x) + self.Wy(self.y) + self.l_tau * self.s * (1 - self.y))
        else:            
            s = F.relu(self.Wx(x) + self.l_tau * self.s * (1 - self.y))
        
        if self.soft:            
            y = F.sigmoid(F.bias(s, self.b))
        else:            
            y = step(F.bias(s, self.b))
            #y = F.relu(F.sign(F.bias(s, self.b)))
            
        self.s = s
        self.y = y
        
        return y