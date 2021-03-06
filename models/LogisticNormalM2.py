"""
This implements a version of the Gumbel Softmax model in:

Jang, Eric, Shixiang Gu, and Ben Poole. "Categorical reparameterization with gumbel-softmax." arXiv preprint arXiv:1611.01144 (2016).

It uses a logistic normal instead of a gumbel softmax.
"""

from GumbelSoftmaxM2 import *

class LogisticNormalM2SemiVAE(GumbelSoftmaxM2SemiVAE):

    def sample_y(self,mu):
        eps = theano.gradient.disconnected_grad(self.srng.normal(mu.shape,dtype=config.floatX))
        if self.params['model']=='LogisticNormalM2':
            y = T.nnet.softmax((eps+mu)*self.tHyperparams['sharpening'])
        elif self.params['model']=='STLogisticNormalM2':
            y = T.nnet.softmax((eps+mu)*self.tHyperparams['sharpening'])
            y_discrete = T.argmax(eps+mu,axis=1)
            y_discrete = T.extra_ops.to_one_hot(y_discrete,self.params['nclasses'],dtype=config.floatX)
            y = theano.gradient.disconnected_grad(y_discrete-y)+y
        else:
            assert False, 'unhandled model type %s' % self.params['model']
        return y

