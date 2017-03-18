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
from vae_ssl_LogGamma import LogGammaSemiVAE
import ipdb


class LogGammaLatentMixtureSemiVAE(LogGammaSemiVAE):

    def __init__(self,params,paramFile=None,reloadFile=None):
        super(LogGammaLatentMixtureSemiVAE,self).__init__(params,paramFile=paramFile,reloadFile=reloadFile)
    def _createParams(self):
        npWeights = super(LogGammaLatentMixtureSemiVAE,self)._createParams()
        K = self.params['nclasses']
        if self.params['learn_prior']:
            npWeights['logbetaprior_W'] = np.zeros((K,K)).astype(config.floatX)
        else:
            npWeights['logbetaprior'] = np.exp(self.params['betaprior']+self.params['sharpening']*np.eye(K)).astype(config.floatX) 
        if self.params['learn_posterior']:
            npWeights['logbeta_y_W'] = np.exp(self.params['betaprior']+np.eye(K)).astype(config.floatX)
        return npWeights


    def _KL_Loggamma(self, beta_q, beta_p):
        """
                                KL Term for LogGamma Variates
        """
        KL = (T.gammaln(beta_p)-T.gammaln(beta_q)-(beta_p-beta_q)*Psi()(beta_q)).sum(1,keepdims=True)
        return KL

    def _KL_logGammaMixture(self,beta_q):
        qy_x = T.nnet.softmax(T.log(beta_q))
        K = self.params['nclasses']
        qylogqy = T.sum(qy_x*T.log(qy_x),axis=1)
        qylogpy = -T.log(self.params['nclasses'])
        KL_Y = qylogqy - qylogpy
        KL_alpha = []
        for y in range(K):
            Y = np.zeros((1,K))
            Y[0,y] = 1.
            if self.params['learn_prior']:
                beta_p = T.exp(self.tWeights['logbetaprior_W'][y]).reshape((1,K))
            else:
                beta_p = T.exp(self.tWeights['logbetaprior'][y]).reshape((1,K))
            if self.params['learn_posterior']:
                beta_y_W = T.exp(self.tWeights['logbeta_y_W'][y]).reshape((1,K))
                beta_q_y = beta_q + beta_y_W
            else:
                beta_q_y = beta_q + Y
            KL_alpha.append(self._KL_Loggamma(beta_q_y,beta_p))
        KL_alpha = T.concatenate(KL_alpha,axis=1)
        KL_alpha = T.sum(qy_x*KL_alpha,axis=1).reshape((-1,1))
        KL_Y = KL_Y.reshape((-1,1))
        return KL_alpha + KL_Y

    def _variationalLoggamma(self,beta_q,betaprior):
        K = self.params['nclasses']
        #beta_p = betaprior*np.ones((1,K))
        KL_alpha = self._KL_logGammaMixture(beta_q)
        loggamma_variates = theano.gradient.disconnected_grad(self.rng_loggamma(beta_q))
        return loggamma_variates, KL_alpha

    def _getModelOutputs(self,outputsU,outputsL,suffix=''):
        my_outputs = super(LogGammaLatentMixtureSemiVAE,self)._getModelOutputs(outputsU,outputsL,suffix)
        if self.params['learn_prior']:
            my_outputs['logbetaprior'] = self.tWeights['logbetaprior_W']
        else:
            my_outputs['logbetaprior'] = self.tWeights['logbetaprior']
        if self.params['learn_posterior']:
            my_outputs['logbeta_y'] = self.tWeights['logbeta_y_W']
        return my_outputs

