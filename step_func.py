# -*- coding: utf-8 -*-

import numpy

from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check

class Step(function_node.FunctionNode):
    """Step function."""
    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        y = utils.force_array(numpy.heaviside(x[0], 0))
        self.retain_outputs((0,))
        self._use_cudnn = False
        return y,

    def forward_gpu(self, x):
        y = cuda.cupy.empty_like(x[0])
        cuda.cupy.maximum(cuda.cupy.sign(x[0]), 0, out=y)
        self._use_cudnn = False

        self.retain_outputs((0,))
        return y,

    def backward(self, indexes, grad_outputs):
        if self._use_cudnn:
            x = self.get_retained_inputs()[0].data
        else:
            x = None
        y = self.get_retained_outputs()[0]
        gy = grad_outputs[0]
        return StepGrad(x).apply((y, gy))


class StepGrad(function_node.FunctionNode):
    """ pseudo-derivative of step as derivative of tanh """
    def __init__(self, x):
        super(StepGrad, self).__init__()
        self.x = x

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1))
        y, gy = inputs
        one = y.dtype.type(1)
        return utils.force_array(gy * (one - y * y)),

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1))
        y, gy = inputs
        gx = cuda.elementwise(
            'T y, T gy', 'T gx',
            'gx = gy * (1 - y * y)',
            'tanh_bwd')(y, gy)
        return gx,

    """
    def backward(self, indexes, grad_outputs):
        y, gy = self.get_retained_inputs()
        ggx = grad_outputs[0]

        y_mul_ggx = y * ggx
        grad_y = -2 * gy * y_mul_ggx
        ggy = ggx - y * y_mul_ggx
        return grad_y, ggy
    """
    
def step(x):
    return Step().apply((x,))[0]