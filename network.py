# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
#import chainer.links as L
from chainer import reporter
from SNU_layer import SNU

#import numpy as xp
from chainer import cuda
xp = cuda.cupy
#from chainer import Variable

# Network definition
class SNU_Network(chainer.Chain):
    def __init__(self, n_in=784, n_mid=256, n_out=10,
                 num_time=100, l_tau=0.8, soft=True, gpu=False, test_mode=False):
        super(SNU_Network, self).__init__()
        with self.init_scope():
            self.l1 = SNU(n_in, n_mid, l_tau=l_tau, soft=soft, gpu=gpu)
            self.l2 = SNU(n_mid, n_mid, l_tau=l_tau, soft=soft, gpu=gpu)
            self.l3 = SNU(n_mid, n_mid, l_tau=l_tau, soft=soft, gpu=gpu)
            self.l4 = SNU(n_mid, n_out, l_tau=l_tau, soft=soft, gpu=gpu)
            
            self.n_out = n_out
            self.num_time = num_time
            self.test_mode = test_mode
    
    def _reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()
        self.l4.reset_state()
        
    def __call__(self, x, y):
        loss = None
        accuracy = None
        sum_out = None
        #sum_out = Variable(xp.zeros((x.shape[0], self.n_out), dtype=self.xp.float32))
        
        self._reset_state()
        
        #if self.test_mode == True:
        #    y_hat_list = []
        
        for t in range(self.num_time):
            x_t = x[:, :, t]
            h = self.l1(x_t)
            h = self.l2(h)
            h = self.l3(h)
            out = self.l4(h)
            
            #if self.test_mode == True:
            #    y_hat_list.append(y_hat)
            sum_out = out if sum_out is None else sum_out + out
            
            #loss_t = F.softmax_cross_entropy(out, y)
        
            loss_t = F.mean_squared_error(out, xp.eye(self.n_out)[y].astype(xp.float32))
            loss = loss_t if loss is None else loss + loss_t

            #sum_out += y_hat
        
        #loss = F.mean_squared_error(sum_out / self.num_time, xp.eye(self.n_out)[y].astype(xp.float32))
        average_out = sum_out/self.num_time
        #loss = F.softmax_cross_entropy(average_out, y)
        #loss = F.mean_squared_error(average_out, xp.eye(self.n_out)[y].astype(xp.float32))
        accuracy = F.accuracy(average_out, y)
        
        reporter.report({'loss': loss}, self)
        reporter.report({'accuracy': accuracy}, self)
        
        #if self.test_mode == True:
        #    return loss, accuracy
        #else:
        return loss
