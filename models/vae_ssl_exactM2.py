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
from utils.optimizer import adam,rmsprop
from utils.misc import saveHDF5
from stats.randomvariates import randomLogGamma
from utils.special import Psi, Polygamma
from theanomodels import BaseModel
from vae_ssl import RelaxedSemiVAE
import ipdb

class ExactSemiVAE(RelaxedSemiVAE):

    def __init__(self,params,paramFile=None,reloadFile=None):
        super(ExactSemiVAE,self).__init__(params,paramFile=paramFile,reloadFile=reloadFile)

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

        #we don't actually use eps here, but do the following to include it in graph computation
        bs = eps.shape[0]
        nc = self.params['nclasses']
        if Y != None:
            nllY = bs*theano.shared(np.log(nc))
            mu, logcov = self._build_qz(Y,hx,evaluation,modifiedBatchNorm,graphprefix)
            self._addVariable(graphprefix+'_mu'+suffix,mu,ignore_warnings=True)
            self._addVariable(graphprefix+'_logcov2'+suffix,logcov,ignore_warnings=True)
            eps = self.srng.normal(mu.shape,0,1,dtype=config.floatX) 
            Z, KL_Z = self._variationalGaussian(mu,logcov,eps)
            #add noise to assist in training
            if add_noise:
                Z = Z + self.srng.normal(Z.shape,0,0.05,dtype=config.floatX)
            #generative model
            _, nllX = self._buildEmission(Y, Z, X, graphprefix, evaluation=evaluation,modifiedBatchNorm=modifiedBatchNorm)
            KL = KL_Z.sum()
            NLL = nllX.sum() + nllY.sum()
        else:
            probs = T.nnet.softmax(logbeta)
            logprobs = T.log(probs)
            KL_Y = T.sum(probs*logprobs,axis=1)+np.log(nc)

            y = [T.extra_ops.to_one_hot(T.extra_ops.repeat(theano.shared(i),repeats=hx.shape[0]),nb_class=nc) for i in range(nc)]
            y = T.concatenate(y,axis=0)
            #hx_repeat = T.extra_ops.repeat(hx,nc,axis=0)
            #X_repeat = T.extra_ops.repeat(X,nc,axis=0)
            hx_repeat = T.tile(hx.T,nc,ndim=2).T
            X_repeat = T.tile(X.T,nc,ndim=2).T

            mu, logcov = self._build_qz(y,hx_repeat,evaluation,modifiedBatchNorm,graphprefix)
            self._addVariable(graphprefix+'_mu'+suffix,mu,ignore_warnings=True)
            self._addVariable(graphprefix+'_logcov2'+suffix,logcov,ignore_warnings=True)
            eps = self.srng.normal(mu.shape,0,1,dtype=config.floatX) 
            Z, KL_Z = self._variationalGaussian(mu,logcov,eps)
            #add noise to assist in training
            if add_noise:
                Z = Z + self.srng.normal(Z.shape,0,0.05,dtype=config.floatX)
            #generative model
            _, nllX = self._buildEmission(y, Z, X_repeat, graphprefix, evaluation=evaluation,modifiedBatchNorm=modifiedBatchNorm)
            
            #evaluate full expectation
            kl = 0
            nll = 0 
            negkl = 0
            #check=  KL_Z.reshape((nc,bs)).T.tag.test_value*probs.tag.test_value
            #could use reshape, but better safe than sorry
            for c in range(nc):
                start_idx = c*bs
                end_idx = (c+1)*bs
                kl += probs[:,c]*KL_Z[start_idx:end_idx].ravel()
                nll += probs[:,c]*nllX[start_idx:end_idx].ravel()
                if not self.params['negKL']:
                    kl += probs[:,c]*KL_Z[start_idx:end_idx].ravel()
                else:
                    kl += theano.gradient.disconnected_grad(probs[:,c])*KL_Z[start_idx:end_idx].ravel()
                    negkl += -2*probs[:,c]*theano.gradient.disconnected_grad(KL_Z[start_idx:end_idx].ravel())
            KL_Z = kl
            nllX = nll
            KL = KL_Y.sum() + KL_Z.sum()
            NLL = nllX.sum() 

        bound = KL + NLL
        #objective function
        if evaluation:
            objfunc = bound
        else:
            if annealKL == None:
                objfunc = bound
            else:
                #annealing
                objfunc = annealKL*KL + NLL
            if Y==None and self.params['negKL']:
                objfunc += negkl.sum()

        outputs = {
                   'Z':Z,
                   'logbeta':logbeta,
                   'bound':bound,
                   'objfunc':objfunc,
                   'nllX':nllX,
                   'KL_Z':KL_Z,
                   'KL':KL,
                   'NLL':NLL}
        if Y!=None:
            outputs['nllY']=nllY
        else:
            outputs['KL_Y']=KL_Y

        return outputs


    def _getModelOutputs(self,outputsU,outputsL,suffix=''):
        my_outputs = {
                         'KL_Y_U':outputsU['KL_Y'].sum(),
                         'nllX_U':outputsU['nllX'].sum(),
                         'NLL_U':outputsU['NLL'].sum(),
                         'KL_U':outputsU['KL'].sum(),
                         'KL_Z_U':outputsU['KL_Z'].sum(),
                         'nllY':outputsL['nllY'].sum(),
                         'nllX_L':outputsL['nllX'].sum(),
                         'NLL_L':outputsL['NLL'].sum(),
                         'KL_L':outputsL['KL'].sum(),
                         'KL_Z_L':outputsL['KL_Z'].sum(),
                         'logbeta_U':outputsU['logbeta'],
                         'logbeta_L':outputsL['logbeta'],
                         'mu_U':self._tVariables['U_mu'+suffix],
                         'logcov2_U':self._tVariables['U_logcov2'+suffix],
                         'mu_L':self._tVariables['L_mu'+suffix],
                         'logcov2_L':self._tVariables['L_logcov2'+suffix]
                         }

        if self.params['bilinear']:
            my_outputs['hz_U'] = self._tVariables['U_hz'+suffix]
            my_outputs['hz_L'] = self._tVariables['L_hz'+suffix]
            my_outputs['hz_W_alpha_U'] = self._tVariables['U_hz_W_alpha'+suffix]
            my_outputs['hz_W_alpha_L'] = self._tVariables['L_hz_W_alpha'+suffix]
            my_outputs['p_Z_embedded_U'] = self._tVariables['U_p_Z_embedded'+suffix]
            my_outputs['p_Z_embedded_L'] = self._tVariables['L_p_Z_embedded'+suffix]
            my_outputs['p_alpha_Z_input_W_U'] = self._tVariables['U_p_alpha_Z_input_W'+suffix]
            my_outputs['p_alpha_Z_input_W_L'] = self._tVariables['L_p_alpha_Z_input_W'+suffix]
        return my_outputs




