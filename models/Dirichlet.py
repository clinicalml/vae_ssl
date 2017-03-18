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


class DirichletSemiVAE(LogGammaSemiVAE):

    def __init__(self,params,paramFile=None,reloadFile=None):
        super(DirichletSemiVAE,self).__init__(params,paramFile=paramFile,reloadFile=reloadFile)

    def _LogBetaFunction(self,beta,**kwargs):
        return T.sum(T.gammaln(beta),**kwargs)-T.gammaln(beta.sum(**kwargs))

    def _DirichletNegCrossEntropy(self,beta_q,beta_p):
        """
            KL divergence D(q||p) 
            where q and p are dirichlet with parameters beta_q and beta_p, respectively
        """
        logB = self._LogBetaFunction(beta_p,axis=1,keepdims=True)
        return -logB + T.sum((beta_p-1)*(Psi()(beta_q)-Psi()(beta_q.sum(axis=1,keepdims=True))),axis=1,keepdims=True)

    def _KL_Dirichlet(self,beta_q,beta_p):
        """
            Compute E_{q(alpha|beta_q)}[logq(alpha|beta_q)-logp(alpha|beta_p)]
        """
        qlogq = self._DirichletNegCrossEntropy(beta_q,beta_q)
        qlogp = self._DirichletNegCrossEntropy(beta_q,beta_p)
        return qlogq-qlogp

    def _logpdf_LogGamma(self, U, beta):
        """
                         log probability density function for Dirichlet
        """
        X = T.nnet.softmax(U)
        return -self._LogBetaFunction(beta,axis=1,keepdims=True) + T.sum((beta-1.)*T.log(X),axis=1,keepdims=True)

    def _variationalLoggamma(self,beta_q,betaprior):
        loggamma_variates = theano.gradient.disconnected_grad(self.rng_loggamma(beta_q))
        K = self.params['nclasses']
        beta_p = betaprior*np.ones((1,K))
        KL_alpha = self._KL_Dirichlet(beta_q,beta_p)
        return loggamma_variates, KL_alpha

