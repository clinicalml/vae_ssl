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
from vae_ssl_LogGamma_LogisticNormal import LogisticNormalSemiVAE
import ipdb


class LogisticNormalMPSemiVAE(LogisticNormalSemiVAE):

    def __init__(self,params,paramFile=None,reloadFile=None):
        super(LogisticNormalMPSemiVAE,self).__init__(params,paramFile=paramFile,reloadFile=reloadFile)

    def _logpdf_Gaussian(self,Z,mu,logcov2):
        K = Z.shape[1]
        return -0.5*(T.log(2*np.pi) + logcov2 + (Z-mu)**2.0/T.exp(logcov2))

    def _qlogp_Gaussian(self,mu_p,s_p,mu_q,s_q):
        '''
        s ~ logcov2
        '''
        return -0.5*(T.log(2*np.pi)+s_p+T.exp(-s_p)*(T.exp(s_q)+(mu_p-mu_q)**2.0))

    def _MixtureGaussianKL(self,mu,logcov2):
        qlogq = T.sum(-0.5*(T.log(2*np.pi)+logcov2+1),axis=1).reshape((-1,1))

        K = self.params['nclasses']
        c = self.params['LogitNormalMP']
        mixture_mu = -c*np.ones((K,K))
        mixture_logcov2 = np.zeros((K,K))
        mixture_mu[np.arange(K),np.arange(K)] = c
        mixture_mu = theano.gradient.disconnected_grad(theano.shared(mixture_mu))
        mixture_logcov2 = theano.gradient.disconnected_grad(theano.shared(mixture_logcov2))

        qlogp = []
        for k in range(K):
            mu_p = mixture_mu[k].reshape((1,-1))
            s_p = mixture_logcov2[k].reshape((1,-1))

            qlogp.append(self._qlogp_Gaussian(mu_p,s_p,mu,logcov2).sum(axis=1).reshape((-1,1)))
        qlogp = T.concatenate(qlogp,axis=1)
        
        KLs = qlogq - qlogp
        minKL = T.min(KLs,axis=1)
        return minKL

        
    def _variationalGaussian2(self, mu, logcov2, eps):
        """
                            KL divergence between N(0,I) and N(mu,exp(logcov))
        """
        #Pass z back
        z, KL = None, None
        if self.params['inference_model']=='single':
            z       = mu + T.exp(0.5*logcov2)*eps
            KL      = self._MixtureGaussianKL(mu,logcov2)
        else:
            assert False,'Invalid inference model: '+self.params['inference_model']
        return z,KL
        
    def _buildGraph(self, X, eps, betaprior=0.2, Y=None, dropout_prob=0.,add_noise=False,annealKL=None,evaluation=False,modifiedBatchNorm=False,graphprefix=None):
        """
                                Build VAE subgraph to do inference and emissions
                (if Y==None, build upper bound of -logp(x), else build upper bound of -logp(x,y)
        """
        if Y == None:
            self._p(('Building graph for lower bound of logp(x)'))
        else:
            self._p(('Building graph for lower bound of logp(x,y)'))
        hx, mu_Z2,logcov2_Z2= self._buildRecognitionBase(X,dropout_prob,evaluation,modifiedBatchNorm,graphprefix)
        suffix = '_e' if evaluation else '_t'
        self._addVariable(graphprefix+'_h(x)'+suffix,hx,ignore_warnings=True)

        if Y!=None:
            mu_Z2 += self._LinearNL(self.tWeights['q_Z2_mu_W|y'],0,Y,onlyLinear=True)
            logcov2_Z2 += self._LinearNL(self.tWeights['q_Z2_logcov2_W|y'],0,Y,onlyLinear=True)
        self._addVariable(graphprefix+'_Z2_mu'+suffix,mu_Z2,ignore_warnings=True)
        self._addVariable(graphprefix+'_Z2_logcov2'+suffix,logcov2_Z2,ignore_warnings=True)
        eps2 = self.srng.normal(mu_Z2.shape,1,dtype=config.floatX)
        Z2, KL_Z2 = self._variationalGaussian2(mu_Z2,logcov2_Z2,eps2)
        if add_noise:
            Z2 = Z2 + self.srng.normal(Z2.shape,0,0.05,dtype=config.floatX)
        if self.params['no_softmax']:
            alpha = Z2
        else:
            if self.params['sharpening'] != 1:
                alpha = T.nnet.softmax(Z2*self.params['sharpening'])
            else:
                alpha = T.nnet.softmax(Z2)
        if Y!=None:
            nllY = T.nnet.categorical_crossentropy(T.nnet.softmax(Z2),Y)
        
        mu, logcov = self._build_qz(alpha,hx,evaluation,modifiedBatchNorm,graphprefix)
        self._addVariable(graphprefix+'_mu'+suffix,mu,ignore_warnings=True)
        self._addVariable(graphprefix+'_logcov2'+suffix,logcov,ignore_warnings=True)
        #Z = gaussian variates
        Z, KL_Z = self._variationalGaussian(mu,logcov,eps)
        if add_noise:
            Z = Z + self.srng.normal(Z.shape,0,0.05,dtype=config.floatX)
        #generative model
        _, nllX = self._buildEmission(alpha, Z, X, graphprefix, evaluation=evaluation,modifiedBatchNorm=modifiedBatchNorm)

        #negative of the lower bound
        if self.params['KL_loggamma_coef'] != 1:
            KL = self.params['KL_loggamma_coef']*KL_Z2.sum() + KL_Z.sum()
        else:
            KL = KL_Z2.sum() + KL_Z.sum()
        NLL = nllX.sum()
        if Y!=None:
            NLL += nllY.sum()
        bound = KL + NLL
        #objective function
        if evaluation:
            objfunc = bound
        else:
            if annealKL == None:
                objfunc = bound
            else:
                objfunc = annealKL*KL + NLL

        outputs = {'alpha':alpha,
                   'Z':Z,
                   'Z2':Z2,
                   'bound':bound,
                   'objfunc':objfunc,
                   'nllX':nllX,
                   'KL_Z2':KL_Z2,
                   'KL_Z':KL_Z,
                   'KL':KL,
                   'NLL':NLL}
        if Y!=None:
            outputs['nllY']=nllY

        return outputs

    def _buildClassifier(self, X, Y, add_noise, dropout_prob, evaluation, modifiedBatchNorm, graphprefix):
        hx, mu_Z2,logcov2_Z2= self._buildRecognitionBase(X,dropout_prob,evaluation,modifiedBatchNorm,graphprefix)
        if evaluation:
            alpha = T.nnet.softmax(mu_Z2)
        else:
            eps2 = self.srng.normal(mu_Z2.shape,1,dtype=config.floatX)
            Z2, KL_Z2 = self._variationalGaussian2(mu_Z2,logcov2_Z2,eps2)
            if add_noise:
                Z2 = Z2 + self.srng.normal(Z2.shape,0,0.05,dtype=config.floatX)
            if self.params['sharpening'] != 1:
                alpha = T.nnet.softmax(Z2*self.params['sharpening'])
            else:
                alpha = T.nnet.softmax(Z2)
        #T.nnet.categorical_crossentropy returns a vector of length batch_size
        probs = alpha
        loss= T.nnet.categorical_crossentropy(probs,Y)
        ncorrect = T.eq(T.argmax(probs,axis=1),Y).sum()
        return probs, loss, ncorrect


