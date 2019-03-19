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
        self.retain_inputs((0,))
        y = utils.force_array(numpy.heaviside(x[0], 0))
        self.retain_outputs((0,))
        self._use_cudnn = False
        return y,

    def forward_gpu(self, x):
        y = cuda.cupy.empty_like(x[0])
        self.retain_inputs((0,))
        cuda.cupy.maximum(cuda.cupy.sign(x[0]), 0, out=y)
        self._use_cudnn = False

        self.retain_outputs((0,))
        return y,

    def backward(self, indexes, grad_outputs):
        x, = self.get_retained_inputs()
        return PseudoStepGrad(x.data).apply(grad_outputs)

class PseudoStepGrad(function_node.FunctionNode):
    def __init__(self, x):
        self.x = x

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('gy',))

        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == self.x.dtype
        )

    def forward_cpu(self, inputs):
        gy, = inputs
        gx = ((-1 < self.x) & (self.x < 1)) * gy * 1
        return utils.force_array(gx, self.x.dtype),

    def forward_gpu(self, inputs):
        gy, = inputs
        return cuda.elementwise(
            'T x, T g', 'T gx',
            'gx = fabs(x) < 0.5 ? 1 * g : 0',
            'step_pseudo_bwd'
        )(self.x, gy),

    """
    def forward_gpu(self, inputs):
        gy, = inputs
        return cuda.elementwise(
            'T x, T g', 'T gx',
            'gx = fabs(x) < 0.5 ? 1 * g : 0',
            'step_pseudo_bwd'
        )(self.x, gy),
    """

    def backward(self, indexes, grad_outputs):
        return PseudoStepGrad(self.x).apply(grad_outputs)
    
def step(x):
    return Step().apply((x,))[0]