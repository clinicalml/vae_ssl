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


class DirichletMixtureSemiVAE(LogGammaSemiVAE):

    def __init__(self,params,paramFile=None,reloadFile=None):
        super(DirichletMixtureSemiVAE,self).__init__(params,paramFile=paramFile,reloadFile=reloadFile)
    def _createParams(self):
        npWeights = super(DirichletMixtureSemiVAE,self)._createParams()
        K = self.params['nclasses']
        npWeights['logbetaprior_W'] = np.zeros((K,K)).astype(config.floatX)
        return npWeights


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

    def _variationalDirichletMixture(self,beta,betaprior,qy_x):
        beta_q = beta 
        K = self.params['nclasses']
        qylogqy = T.sum(qy_x*T.log(qy_x),axis=1)
        qylogpy = -T.log(self.params['nclasses'])
        KL_Y = qylogqy - qylogpy
        KL_alpha = []
        for y in range(K):
            Y = np.zeros((1,K))
            Y[0,y] = 1.
            #beta_p = betaprior*np.ones((1,K)) + Y
            #beta_p = T.sum(Y.reshape(list(Y.shape)+[1])*self.tWeights['betaprior_W'].reshape((1,K,K)),axis=1)
            betaprior_W = T.exp(self.tWeights['logbetaprior_W'].reshape((1,K,K)))
            beta_p = T.sum(Y.reshape(list(Y.shape)+[1])*betaprior_W,axis=1)
            KL_alpha.append(self._KL_Dirichlet(beta_q+Y,beta_p))
        KL_alpha = T.concatenate(KL_alpha,axis=1)
        KL_alpha = T.sum(qy_x*KL_alpha,axis=1)
        loggamma_variates = theano.gradient.disconnected_grad(self.rng_loggamma(beta_q))
        return loggamma_variates, KL_alpha, KL_Y

    def _variationalDirichlet(self,beta_q,beta_p):
        K = self.params['nclasses']
        KL_alpha = self._KL_Dirichlet(beta_q,beta_p)
        loggamma_variates = theano.gradient.disconnected_grad(self.rng_loggamma(beta_q))
        return loggamma_variates, KL_alpha

    def _buildGraph(self, X, eps, betaprior=0.2, Y=None, dropout_prob=0.,add_noise=False,annealKL=None,evaluation=False,modifiedBatchNorm=False,graphprefix=None):
        """
                                Build VAE subgraph to do inference and emissions 
                (if Y==None, build upper bound of -logp(x), else build upper bound of -logp(x,y)
        """
        if Y == None:
            self._p(('Building graph for lower bound of logp(x)'))
        else:
            self._p(('Building graph for lower bound of logp(x,y)'))
        hx, logbeta = self._buildRecognitionBase(X,dropout_prob,evaluation,modifiedBatchNorm,graphprefix)
        suffix = '_e' if evaluation else '_t'
        self._addVariable(graphprefix+'_h(x)'+suffix,hx,ignore_warnings=True)

        # inference on alpha
        beta = T.exp(logbeta)
        bs = eps.shape[0]
        K = self.params['nclasses']
        if Y is None:
            qy_x = T.nnet.softmax(logbeta)
            U, KL_alpha, KL_Y = self._variationalDirichletMixture(beta,betaprior,qy_x)
        else:
            nllY = bs*theano.shared(np.log(K))
            beta += Y
            #beta_p = betaprior*np.ones((1,K)) + self.params['sharpening']*Y
            betaprior_W = T.exp(self.tWeights['logbetaprior_W'].reshape((1,K,K)))
            beta_p = T.sum(Y.reshape(list(Y.shape)+[1])*betaprior_W,axis=1)
            U, KL_alpha = self._variationalDirichlet(beta,beta_p)
        alpha = T.nnet.softmax(U)

        # inference on Z
        mu, logcov = self._build_qz(alpha,hx,evaluation,modifiedBatchNorm,graphprefix) 
        self._addVariable(graphprefix+'_mu'+suffix,mu,ignore_warnings=True)
        self._addVariable(graphprefix+'_logcov2'+suffix,logcov,ignore_warnings=True)
        #Z = gaussian variates
        Z, KL_Z = self._variationalGaussian(mu,logcov,eps)
        if add_noise:
            Z = Z + self.srng.normal(Z.shape,0,0.05,dtype=config.floatX)
        #generative model
        paramsX, nllX = self._buildEmission(alpha, Z, X, graphprefix, evaluation=evaluation,modifiedBatchNorm=modifiedBatchNorm)

        #negative of the lower bound
        if self.params['KL_loggamma_coef'] != 1:
            assert False, "should not be here"
        KL = KL_alpha.sum() + KL_Z.sum()
        NLL = nllX.sum()
        if Y is None:
            KL += KL_Y.sum()
        else:
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

        #gradient hack to do black box variational inference:
        if evaluation == False:
            # make sure sizes are correct to prevent unintentional broadcasting
            KL_Z = KL_Z.reshape([-1])
            nllX = nllX.reshape([-1])
            if self.params['negKL']:
                #negative KL trick :(
                f = theano.gradient.disconnected_grad(-2.*KL_Z+nllX)
            else:
                if annealKL!=None:
                    f = theano.gradient.disconnected_grad(annealKL*KL_Z+nllX)
                else:
                    f = theano.gradient.disconnected_grad(KL_Z+nllX)
            BBVIgradientHack = f*self._logpdf_LogGamma(U,beta).reshape([-1])
            objfunc += BBVIgradientHack.sum()

        outputs = {'alpha':alpha,
                   'Z':Z,
                   'paramsX':paramsX[0],
                   'logbeta':logbeta,
                   'bound':bound,
                   'objfunc':objfunc,
                   'nllX':nllX,
                   'KL_alpha':KL_alpha,
                   'KL_Z':KL_Z,
                   'KL':KL,
                   'NLL':NLL}
        if Y is None:
            outputs['KL_Y']=KL_Y
        else:
            outputs['nllY']=nllY
            outputs['Y']=Y

        return outputs

    def _getModelOutputs(self,outputsU,outputsL,suffix=''):
        my_outputs = {
                         'KL_alphaU':outputsU['KL_alpha'],
                         'KL_Y':outputsU['KL_Y'],
                         'nllX_U':outputsU['nllX'],
                         #'pX_U':outputsU['paramsX'],
                         'NLL_U':outputsU['NLL'].sum(),
                         'KL_U':outputsU['KL'].sum(),
                         'KL_Z_U':outputsU['KL_Z'].sum(),
                         'nllY':outputsL['nllY'].sum(),
                         'Y':outputsL['Y'],
                         'KL_alphaL':outputsL['KL_alpha'],
                         'nllX_L':outputsL['nllX'],
                         #'pX_L':outputsL['paramsX'],
                         'NLL_L':outputsL['NLL'].sum(),
                         'KL_L':outputsL['KL'].sum(),
                         'KL_Z_L':outputsL['KL_Z'].sum(),
                         'logbeta_U':outputsU['logbeta'],
                         'logbeta_L':outputsL['logbeta'],
                         'mu_U':self._tVariables['U_mu'+suffix],
                         'mu_L':self._tVariables['L_mu'+suffix],
                         'logcov2_U':self._tVariables['U_logcov2'+suffix],
                         'logcov2_L':self._tVariables['L_logcov2'+suffix],
                         'logbetaprior':self.tWeights['logbetaprior_W'],
                         }
        return my_outputs

