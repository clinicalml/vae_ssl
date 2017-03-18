import six.moves.cPickle as pickle
from collections import OrderedDict
import sys, time, os
import numpy as np
import gzip, warnings
import theano
from theano import config
theano.config.compute_test_value = 'warn'
from theano.compile.ops import as_op
from theano.printing import pydotprint
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theanomodels.utils.optimizer import adam,rmsprop
from theanomodels.utils.misc import saveHDF5
from randomvariates import randomLogGamma
from special import Psi, Polygamma
from theanomodels.models import BaseModel
from GumbelSoftmaxM2 import SemiVAE 
import ipdb

class LogisticNormalM2(SemiVAE):

    def __init__(self,params,paramFile=None,reloadFile=None):
        super(LogisticNormalM2,self).__init__(params,paramFile=paramFile,reloadFile=reloadFile)

    def _sample_Y(self,logprobs):
        g = theano.gradient.disconnected_grad(self.srng.normal(logprobs.shape,dtype=config.floatX))
        if self.params['model']=='LogisticNormalM2':
            y = T.nnet.softmax((g+logprobs)*self.tWeights['sharpening'])
        elif self.params['model']=='STLogisticNormalM2':
            y = T.nnet.softmax((g+logbeta)*self.tWeights['sharpening'])
            y_discrete = T.argmax(g+logbeta,axis=1)
            y_discrete = T.extra_ops.to_one_hot(y_discrete,self.params['nclasses'],dtype=config.floatX)
            y = theano.gradient.disconnected_grad(y_discrete-y)+y
        else:
            assert False, 'unhandled model type %s' % self.params['model']
        return y

