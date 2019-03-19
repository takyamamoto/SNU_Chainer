# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
#import chainer.links as L
from chainer import reporter
from . import snu_layer

#import numpy as xp
from chainer import cuda
xp = cuda.cupy
#from chainer import Variable

# Network definition
class SNU_Network(chainer.Chain):
    def __init__(self, n_in=784, n_mid=256, n_out=10,
                 num_time=20, l_tau=0.8, soft=False, gpu=False,
                 test_mode=False):
        super(SNU_Network, self).__init__()
        with self.init_scope():
            self.l1 = snu_layer.SNU(n_in, n_mid, l_tau=l_tau, soft=soft, gpu=gpu)
            self.l2 = snu_layer.SNU(n_mid, n_mid, l_tau=l_tau, soft=soft, gpu=gpu)
            self.l3 = snu_layer.SNU(n_mid, n_out, l_tau=l_tau, soft=soft, gpu=gpu)
            
            self.n_out = n_out
            self.num_time = num_time
            self.test_mode = test_mode
    
    def _reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()
        
    def forward(self, x, y):
        loss = None
        accuracy = None
        sum_out = None
        
        self._reset_state()
        
        if self.test_mode == True:
            h1_list = []
            h2_list = []
            out_list = []
        
        for t in range(self.num_time):
            x_t = x[:, :, t]
            h1 = self.l1(x_t)
            h2 = self.l2(h1)
            out = self.l3(h2)
            
            if self.test_mode == True:
                h1_list.append(h1)
                h2_list.append(h2)
                out_list.append(out)
            
            sum_out = out if sum_out is None else sum_out + out
        
        loss = F.softmax_cross_entropy(sum_out, y)
        accuracy = F.accuracy(sum_out, y)

        reporter.report({'loss': loss}, self)
        reporter.report({'accuracy': accuracy}, self)
        
        if self.test_mode == True:
            return loss, accuracy, h1_list, h2_list, out_list
        else:
            return loss
