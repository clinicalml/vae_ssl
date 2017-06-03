from GumbelSoftmaxM2 import *

class LogisticNormalM2SemiVAE(GumbelSoftmaxM2SemiVAE):

    def _sample_Y(self,mu):
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

