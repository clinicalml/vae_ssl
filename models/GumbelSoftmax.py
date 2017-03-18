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
from LogGamma import LogGammaSemiVAE
import scipy
import ipdb


def logsumexp(logs):
    maxlog = T.max(logs,axis=1,keepdims=True)
    return maxlog + T.log(T.sum(T.exp(logs-maxlog),axis=1,keepdims=True))

class GumbelSoftmax(LogGammaSemiVAE):

    def __init__(self,params,paramFile=None,reloadFile=None):
        super(GumbelSoftmax,self).__init__(params,paramFile=paramFile,reloadFile=reloadFile)

    def _sample_Y(self,g):
        if self.params['model']=='GumbelSoftmax':
            y = T.nnet.softmax(g*self.tWeights['sharpening'])
        elif self.params['model']=='STGumbelSoftmax':
            y = T.nnet.softmax(g*self.tWeights['sharpening'])
            y_discrete = T.argmax(g,axis=1)
            y_discrete = T.extra_ops.to_one_hot(y_discrete,self.params['nclasses'],dtype=config.floatX)
            y = theano.gradient.disconnected_grad(y_discrete-y)+y
        else:
            assert False, 'unhandled GumbelSoftmax type %s' % self.params['model']
        return y

    def _variationalGumbel(self,mu_q,mu_p=0):
        mu_diff = mu_q - mu_p
        KL = T.sum(-1. + mu_diff + T.exp(-mu_diff),axis=1)
        u = self.srng.uniform(mu_q.shape,low=1e-10,high=1.-1e-10,dtype=config.floatX)
        u = theano.gradient.disconnected_grad(u)
        g = mu_q + -T.log(-T.log(u))
        return g, KL

    def _buildGraph(self, X, eps, betaprior=0.2, Y=None, dropout_prob=0.,add_noise=False,annealKL_Z=None,annealKL_alpha=None,evaluation=False,modifiedBatchNorm=False,graphprefix=None):
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

        mu_Z2 = logbeta
        if Y is not None:
            if self.params['learn_posterior']:
                mu_Z2 = mu_Z2 + T.nnet.softplus(self.tWeights['posterior_W'][0])*Y
            else:
                mu_Z2 = mu_Z2 + T.nnet.softplus(self.params['posterior_c'])*Y
        self._addVariable(graphprefix+'_Z2_mu'+suffix,mu_Z2,ignore_warnings=True)
        Z2, KL_Z2 = self._variationalGumbel(mu_Z2)
        alpha = self._sample_Y(Z2)

        if Y is not None:
            nllY = T.nnet.categorical_crossentropy(alpha,Y)
        
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
        NLL = nllX.sum()
        if Y is not None:
            NLL += nllY.sum()
        KL = KL_Z.sum() + KL_Z2.sum()
        bound = KL + NLL
        #objective function
        if evaluation:
            objfunc = bound
        else:
            _KL_Z = KL_Z.sum()
            _KL_alpha = KL_Z2.sum()
            if annealKL_Z is not None:
                _KL_Z = annealKL_Z*_KL_Z
            if annealKL_alpha is not None:
                _KL_alpha = annealKL_alpha*_KL_alpha
            objfunc = _KL_Z + _KL_alpha + NLL

        outputs = {'alpha':alpha,
                   'Z':Z,
                   'Z2':Z2,
                   'mu_Z2':mu_Z2,
                   'logbeta':logbeta,
                   'bound':bound,
                   'objfunc':objfunc,
                   'nllX':nllX,
                   'KL_Z2':KL_Z2,
                   'KL_Z':KL_Z,
                   'KL':KL,
                   'NLL':NLL}
        if Y!=None:
            outputs['nllY']=nllY
            outputs['Y']=Y

        return outputs

    def _getModelOutputs(self,outputsU,outputsL,suffix=''):
        my_outputs = {
                         'KL_Z2_U':outputsU['KL_Z2'].sum(),
                         'nllX_U':outputsU['nllX'].sum(),
                         'NLL_U':outputsU['NLL'].sum(),
                         'KL_U':outputsU['KL'].sum(),
                         'KL_Z_U':outputsU['KL_Z'].sum(),
                         'nllY':outputsL['nllY'].sum(),
                         'KL_Z2_L':outputsL['KL_Z2'].sum(),
                         'nllX_L':outputsL['nllX'].sum(),
                         'NLL_L':outputsL['NLL'].sum(),
                         'KL_L':outputsL['KL'].sum(),
                         'KL_Z_L':outputsL['KL_Z'].sum(),
                         'mu_Z2_U':self._tVariables['U_Z2_mu'+suffix],
                         'mu_Z2_L':self._tVariables['L_Z2_mu'+suffix],
                         'mu_U':self._tVariables['U_mu'+suffix],
                         'mu_L':self._tVariables['L_mu'+suffix],
                         'logcov2_U':self._tVariables['U_logcov2'+suffix],
                         'logcov2_L':self._tVariables['L_logcov2'+suffix],
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


