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


def logsumexp(logs):
    maxlog = T.max(logs,axis=1,keepdims=True)
    return maxlog + T.log(T.sum(T.exp(logs-maxlog),axis=1,keepdims=True))

def DirichletNegEntropy(beta):
    K = beta.shape[1]
    betasum = beta.sum(axis=1,keepdims=True)
    logB = T.gammaln(beta).sum(axis=1,keepdims=True)-T.gammaln(betasum)
    return -logB + (K-betasum)*Psi()(betasum) + T.sum((beta-1)*Psi()(beta),axis=1,keepdims=True)

class MixDirPriorSemiVAE(LogGammaSemiVAE):

    def __init__(self,params,paramFile=None,reloadFile=None):
        super(MixDirPriorSemiVAE,self).__init__(params,paramFile=paramFile,reloadFile=reloadFile)

    def _LoggammaKL(self,beta,betaprior):
        K = self.params['nclasses'] 
        betamax = self.params['betamax']
        betamin = betaprior
        betapriors = betamin*np.ones(K)
        betapriors[0] = betamax
        
        qlogq = DirichletNegEntropy(beta).ravel()
        
        S = self.params['logp_S']
        logps = []
        stime = time.time()
        for s in range(S):
            loggamma_variates = theano.gradient.disconnected_grad(self.rng_loggamma(beta))
            alpha = T.nnet.softmax(loggamma_variates)
            logalpha = T.log(alpha)
            logps.append(logsumexp((betamax-betamin)*logalpha).reshape((-1,1)))
        print (time.time()-stime)
        logp = T.concatenate(logps,axis=1).mean(axis=1).ravel()
        logB = T.gammaln(betapriors).sum()-T.gammaln(betapriors.sum())
        qlogp = logp - logB.ravel() - (betamin-1)*K*Psi()(beta.sum(axis=1)).ravel() + (betamin-1)*T.sum(Psi()(beta),axis=1).ravel() - T.log(K)
        
        BBVIgradientHack = theano.gradient.disconnected_grad(logp)*self._logpdf_LogGamma(loggamma_variates,beta).ravel()
        self._BBVIgradientHack_loggamma = BBVIgradientHack

        KL = qlogq.ravel()-qlogp.ravel()
        return KL

    def _buildGraph(self, X, eps, betaprior=0.2, Y=None, dropout_prob=0.,add_noise=False,annealKL=None,evaluation=False,modifiedBatchNorm=False,graphprefix=None):
        kwargs = {k:v for k,v in locals().iteritems() if k!='self'}
        outputs = super(MixDirPriorSemiVAE,self)._buildGraph(**kwargs)
        if evaluation == False:
            assert self.params['logpxy_discrete']==False, 'unhandled setting: logpxy_discrete'
            outputs['objfunc'] += self._BBVIgradientHack_loggamma.sum()
        return outputs

