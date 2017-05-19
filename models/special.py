"""
Author: Justin Mao-Jones
Special Theano Ops for incorporating functions in scipy.special not yet available in Theano codebase.
Template derived from http://deeplearning.net/software/theano/extending/extending_theano.html
"""

import theano
import scipy.special

class Polygamma(theano.Op):
    __props__ = ("k",)

    def __init__(self, k):
        self.k = k
        super(Polygamma, self).__init__()
        
    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        # Note: using x_.type() is dangerous, as it copies x's broadcasting
        # behaviour
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x, = inputs
        z, = output_storage
        z[0] = scipy.special.polygamma(self.k,x)

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

    def grad(self, inputs, output_grads):
        x, = inputs
        return [output_grads[0] * Polygamma(self.k+1)(x)]

    def R_op(self, inputs, eval_points):
        # R_op can receive None as eval_points.
        # That mean there is no diferientiable path through that input
        # If this imply that you cannot compute some outputs,
        # return None for those.
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

class Psi(theano.Op):
    __props__ = ()

    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        # Note: using x_.type() is dangerous, as it copies x's broadcasting
        # behaviour
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x, = inputs
        z, = output_storage
        z[0] = scipy.special.digamma(x)

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

    def grad(self, inputs, output_grads):
        x, = inputs
        return [output_grads[0] * Polygamma(1)(x)]

    def R_op(self, inputs, eval_points):
        # R_op can receive None as eval_points.
        # That mean there is no diferientiable path through that input
        # If this imply that you cannot compute some outputs,
        # return None for those.
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)


if __name__ == '__main__':
    import numpy as np

    print 'check Psi()'
    x = theano.tensor.vector('X')
    y = Psi()(x)
    out = y.sum()
    g = theano.tensor.grad(out,x)
    f = theano.function([x],[out,g.sum()],allow_input_downcast=True )

    a = np.exp(np.arange(-5,5,0.2))
    y, g = f(a)
    print 'y = %s' % y
    print 'y-scipy.special.psi(a).sum() = %s' % (y-scipy.special.psi(a).sum())
    print 'g = %s' % g
    print 'g-scipy.special.polygamma(1,a).sum() = %s' % (g-scipy.special.polygamma(1,a).sum())

    print 'check Polygamma(1)'
    x = theano.tensor.vector('X')
    y = Polygamma(1)(x)
    out = y.sum()
    g = theano.tensor.grad(out,x)
    f = theano.function([x],[out,g.sum()],allow_input_downcast=True )

    a = np.exp(np.arange(-5,5,0.2))
    y, g = f(a)
    print 'y = %s' % y
    print 'y-scipy.special.polygamma(1,a).sum() = %s' % (y-scipy.special.polygamma(1,a).sum())
    print 'g = %s' % g
    print 'g-scipy.special.polygamma(2,a).sum() = %s' % (g-scipy.special.polygamma(2,a).sum())

