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
from __init__ import BaseModel
import ipdb
import random

@as_op(itypes=[T.fmatrix,T.fscalar],otypes=[T.fmatrix])
def rng_loggamma_(beta,seed):
    vfunc = np.vectorize(randomLogGamma)
    random.seed(float(seed))
    return vfunc(beta).astype('float32')


class RelaxedSemiVAE(BaseModel, object):
    def __init__(self, params, paramFile=None, reloadFile=None):
        self._tVariables = {}
        self._BNprefix = None
        super(RelaxedSemiVAE,self).__init__(params, paramFile=paramFile, reloadFile=reloadFile)

    def rng_loggamma(self, beta):
        seed=self.srng.uniform(size=(1,),low=-1.0e10,high=1.0e10)[0]
        return rng_loggamma_(beta,seed)

    def _countParams(self,params=None):
        """
        _countParams: Count the number of parameters in the model that will be optimized
        """
        if params==None:
            self.nParams    = 0
            for k in self.npWeights:
                ctr = np.array(self.npWeights[k].shape).prod()
                self.nParams+= ctr
            self._p(('Nparameters: %d')%self.nParams)
            return self.nParams
        else:
            nParams    = 0
            for p in params:
                ctr = np.array(p.get_value().shape).prod()
                nParams+= ctr
            return nParams

    def _addVariable(self, name, data, ignore_warnings=False):
        """
        Add to tVariables
        """
        if name not in self._tVariables:
            self._p('Adding %s to self._tVariables' % name)
            self._tVariables[name] = data
        else:
            if not ignore_warnings:
                warnings.warn(name+" found in tVariables. No action taken")	

    def _addWeights(self, name, data, ignore_warnings=False, **kwargs):
        """
        Add to npWeights/tWeights
        If you would like it to be updated (taken gradients with respect to)
        Make *sure* you have U_ _U, W_ _W b_ _b as part of the name
        """
        if name not in self.npWeights:
            plist = ['_U','U_','W_','_W','_b','b_']
            if not np.any([k in name for k in plist]):
                self._p('WARNING: '+name+' will not differentiated with respect to')
            self.npWeights[name] = data.astype(config.floatX)
            self.tWeights[name]  = theano.shared(self.npWeights[name], name=name,**kwargs)
        else:
            if not ignore_warnings:
                warnings.warn(name+" found in npWeights. No action taken")

    def _addUpdate(self, var, data, ignore_warnings=False):
        """
        Add an update for tWeights
        """
        if len(self.updates) > 0 and var in zip(*self.updates)[0]:
            if not ignore_warnings:
                warnings.warn(var.name+' found in self.updates...no action taken')
        else:
            self.updates.append((var,data))

    def _getModelParams(self, restrict = ''):
        """
        Return list of model parameters to take derivatives with respect to
        """
        paramlist = []
        namelist  = []
        otherparamnames = []
        for k in self.tWeights.values():
            if 'W_' in k.name or 'b_' in k.name or '_b' in k.name or '_W' in k.name or 'U_' in k.name or '_U' in k.name:
                #Use this to only get a list of parameters with specific substrings like 'p_'
                #Since it is set to '' by default, it should appear in all strings
                if restrict in k.name:
                    paramlist.append(k)
                    namelist.append(k.name)
        othernames = [k.name for k in self.tWeights.values() if k.name not in namelist]

        self._p('Params to optimize:\n' + '\n'.join(namelist))
        self._p('Other params:\n' + '\n'.join(othernames))
        return paramlist

    def _createParams(self):
        """
                    _createParams: create parameters necessary for the model
        """
        npWeights = OrderedDict()
        if 'q_dim_hidden' not in self.params or 'p_dim_hidden' not in self.params:
            self.params['q_dim_hidden']= dim_hidden
            self.params['p_dim_hidden']= dim_hidden
        DIM_HIDDEN = self.params['q_dim_hidden']
        #Weights in recognition network model
        for q_l in range(self.params['q_layers']):
            dim_input     = DIM_HIDDEN
            dim_output    = DIM_HIDDEN
            if q_l==0:
                dim_input     = self.params['dim_observations']
            if self.params['nonlinearity']=='maxout':
                dim_output= DIM_HIDDEN*self.params['maxout_stride']
            # h(x)
            npWeights['q_h(x)_'+str(q_l)+'_W'] = self._getWeight((dim_input, dim_output))
            npWeights['q_h(x)_'+str(q_l)+'_b'] = self._getWeight((dim_output, ))
        for a_l in range(self.params['alpha_inference_layers']):
            dim_input     = DIM_HIDDEN
            dim_output    = DIM_HIDDEN
            if self.params['nonlinearity']=='maxout':
                dim_output= DIM_HIDDEN*self.params['maxout_stride']
            # log(beta)
            npWeights['q_logbeta_'+str(a_l)+'_W'] = self._getWeight((dim_input, dim_output))
            npWeights['q_logbeta_'+str(a_l)+'_b'] = self._getWeight((dim_output, ))
        for hz_l in range(self.params['hz_inference_layers']):
            dim_input = DIM_HIDDEN
            dim_output = DIM_HIDDEN
            if self.params['nonlinearity']=='maxout':
                dim_output = DIM_HIDDEN*self.params['maxout_stride']
            npWeights['q_hz_%s_W' % hz_l] = self._getWeight((dim_input, dim_output))
            npWeights['q_hz_%s_b' % hz_l] = self._getWeight((dim_output, ))
        for z_l in range(self.params['z_inference_layers']):
            dim_input     = DIM_HIDDEN
            if z_l == 0 and not self.params['bilinear']:
                dim_input = 2*DIM_HIDDEN
            dim_output    = DIM_HIDDEN
            if self.params['nonlinearity']=='maxout':
                dim_output= DIM_HIDDEN*self.params['maxout_stride']
            # Z 
            npWeights['q_Z_'+str(z_l)+'_W'] = self._getWeight((dim_input, dim_output))
            npWeights['q_Z_'+str(z_l)+'_b'] = self._getWeight((dim_output, ))

        if self.params['inference_model']=='single':
            npWeights['q_logbeta_out_W'] = self._getWeight((DIM_HIDDEN, self.params['nclasses']))
            npWeights['q_logbeta_out_b'] = self._getWeight((self.params['nclasses'],))
            dim_output = DIM_HIDDEN
            if self.params['nonlinearity']=='maxout':
                dim_output= DIM_HIDDEN*self.params['maxout_stride']
            if self.params['bilinear']:
                npWeights['q_alpha_h(x)_W'] = self._getWeight((DIM_HIDDEN,DIM_HIDDEN,self.params['nclasses']))
            else:
                npWeights['q_alpha_h(x)_W'] = self._getWeight((self.params['nclasses'],DIM_HIDDEN))
            npWeights['q_alpha_h(x)_b'] = self._getWeight((DIM_HIDDEN,))
            #npWeights['q_Z_hx_alpha_W'] = self._getWeight((self.params['nclasses']+DIM_HIDDEN,dim_output))
            #npWeights['q_Z_hx_alpha_b'] = self._getWeight((dim_output,))
            npWeights['q_Z_mu_W']     = self._getWeight((DIM_HIDDEN, self.params['dim_stochastic']))
            npWeights['q_Z_mu_b']     = self._getWeight((self.params['dim_stochastic'],))
            npWeights['q_Z_logcov_W'] = self._getWeight((DIM_HIDDEN, self.params['dim_stochastic']))
            npWeights['q_Z_logcov_b'] = self._getWeight((self.params['dim_stochastic'],))

        else:
            assert False,'Invalid variational model'


        #Generative Model
        DIM_HIDDEN = self.params['p_dim_hidden']
        dim_input = self.params['dim_stochastic']
        for pz_l in range(self.params['z_generative_layers']):
            dim_output = DIM_HIDDEN
            if self.params['nonlinearity']=='maxout':
                dim_output = DIM_HIDDEN*self.params['maxout_stride']
            npWeights['p_z_%s_W' % pz_l] = self._getWeight((dim_input, dim_output))
            npWeights['p_z_%s_b' % pz_l] = self._getWeight((dim_output, ))
            dim_input = DIM_HIDDEN
        if self.params['bilinear']:
            npWeights['p_alpha_Z_input_W'] = self._getWeight((DIM_HIDDEN,dim_input,self.params['nclasses']))
            dim_input = DIM_HIDDEN
        else:
            dim_input += self.params['nclasses']
        for p_l in range(self.params['p_layers']):
            dim_output    = DIM_HIDDEN
            if self.params['nonlinearity']=='maxout':
                dim_output= DIM_HIDDEN*self.params['maxout_stride']
            npWeights['p_'+str(p_l)+'_W'] = self._getWeight((dim_input, dim_output))
            npWeights['p_'+str(p_l)+'_b'] = self._getWeight((dim_output, ))
            dim_input     = DIM_HIDDEN
        if self.params['data_type']=='real':
            npWeights['p_mu_W']     = self._getWeight((DIM_HIDDEN, self.params['dim_observations']))
            npWeights['p_logcov_W'] = self._getWeight((DIM_HIDDEN, self.params['dim_observations']))
            npWeights['p_mu_b']     = self._getWeight((self.params['dim_observations'],))
            npWeights['p_logcov_b'] = self._getWeight((self.params['dim_observations'],))
        else:
            npWeights['p_mean_W']     = self._getWeight((DIM_HIDDEN, self.params['dim_observations']))
            npWeights['p_mean_b']     = self._getWeight((self.params['dim_observations'],))
        return npWeights
    
    def _fakeData(self,XU,XL,Y,epsU,epsL):
        """
                                Compile all the fake data 
        """
        XU.tag.test_value = np.random.randint(0,2,(2, self.params['dim_observations'])).astype(config.floatX)
        XL.tag.test_value = np.random.randint(0,2,(20, self.params['dim_observations'])).astype(config.floatX)
        Y.tag.test_value = np.mod(np.arange(20),10).astype('int32')
        epsU.tag.test_value = np.random.randn(2, self.params['dim_stochastic']).astype(config.floatX)
        epsL.tag.test_value = np.random.randn(20, self.params['dim_stochastic']).astype(config.floatX)
    

    def _BNlayer(self, W, b, inp, prefix, onlyLinear=False, evaluation=False, convolution=False):
        """
                                Batch normalization layer
        Based on the implementation from: https://github.com/shuuki4/Batch-Normalization
        a) Support for 3D tensors (for time series data) - here, the last layer is the dimension of the data
        b) Create gamma/beta, create updates for gamma/beta
        c) Different for train/evaluation.
        d) Different for convolutional layers
        """
        W_name     = W.name
        W_shape    = self.npWeights[W_name].shape
        assert len(W_shape)==2,'Expecting W to be a matrix: '+str(len(W_shape))
        gamma_name = prefix+'_W_BN_gamma'
        if gamma_name not in self.tWeights.keys():
            gamma_init = self._getWeight((W_shape[1],))
            #gamma_init = np.ones((W_shape[1],),dtype=config.floatX)
            self._addWeights(gamma_name, gamma_init, borrow=True)
        beta_name  = prefix+'_b_BN_beta'
        if beta_name not in self.tWeights.keys():
            beta_init = self._getWeight((W_shape[1],))
            #beta_init  = np.zeros((W_shape[1],),dtype=config.floatX)
            self._addWeights(beta_name,  beta_init,  borrow=True)
        #Create a running mean that will not be differentiated
        if self._BNprefix != None:
            mean_name  = self._BNprefix + '-' + prefix+'_BN_running_mean'
            var_name   = self._BNprefix + '-' + prefix+'_BN_running_var'
        else:
            mean_name  = prefix+'_BN_running_mean'
            var_name   = prefix+'_BN_running_var'
        if mean_name not in self.tWeights.keys():
            mean_init  = np.zeros((W_shape[1],), dtype=config.floatX)
            self._addWeights(mean_name, mean_init, borrow=True)
        if var_name not in self.tWeights.keys():
            var_init   = np.ones((W_shape[1],),  dtype=config.floatX)
            self._addWeights(var_name,  var_init, borrow=True)
        if 'BN_momentum' not in self.tWeights.keys():
            self._addWeights('BN_momentum', np.asarray(0.,dtype=config.floatX), borrow=True)
        momentum,eps= 0.95, 1e-3
        lin = T.dot(inp,W)+b
        if convolution:
            assert False,'Not implemented'
        else:
            if evaluation:
                normalized = (lin-self.tWeights[mean_name])/T.sqrt(self.tWeights[var_name]+eps)
                bn_lin     = self.tWeights[gamma_name]*normalized + self.tWeights[beta_name]
            else:
                if lin.ndim==2:
                    cur_mean   = lin.mean(0)
                    cur_var    = lin.var(0)
                elif lin.ndim==3:
                    #For now, normalizing across time
                    cur_mean   = lin.mean((0,1))
                    cur_var    = lin.var((0,1))
                else:
                    assert False,'No support for tensors greater than 3D'
                if self._BNprefix != None:
                    batch_mean_name = self._BNprefix + '-' + prefix + '_BN_batch_mean'
                    batch_var_name = self._BNprefix + '-' + prefix + '_BN_batch_var'
                else:
                    batch_mean_name = prefix + '_BN_batch_mean'
                    batch_var_name = prefix + '_BN_batch_var'
                self._addVariable(batch_mean_name,cur_mean)
                self._addVariable(batch_var_name,cur_var)
                normalized     = (lin-cur_mean) / T.sqrt(cur_var+eps)
                bn_lin         = self.tWeights[gamma_name]*normalized + self.tWeights[beta_name]
                #Update running stats
                self._addUpdate(self.tWeights[mean_name], self.tWeights['BN_momentum'] * self.tWeights[mean_name] + (1.0-self.tWeights['BN_momentum']) * cur_mean)
                self._addUpdate(self.tWeights[var_name],  self.tWeights['BN_momentum'] * self.tWeights[var_name] + (1.0-self.tWeights['BN_momentum']) * (float(W_shape[0])/float(W_shape[0]-1))* cur_var)
                #momentum will be 0 in the first iteration
                self._addUpdate(self.tWeights['BN_momentum'],momentum,ignore_warnings=True)
        #Elementwise nonlinearity
        lin_out = bn_lin
        if onlyLinear:
            return lin_out
        else:
            self._p('Adding %s after batchnorm %s'%(self.params['nonlinearity'],W_name))
            return self._applyNL(lin_out,W)
        
    def _BNlayerModified(self, inp, W, b, prefix, onlyLinear=False, evaluation=False):
        """
                Uses the batchnorm means and vars from another batchnorm transformation
        """
        #assert self.params['separateBNrunningstats']==False,"cannot use modified batchnorm with separateBNrunningstats"
        eps = 1e-3
        if evaluation:
            bn_mean = self.tWeights[prefix+'_BN_running_mean']
            bn_var = self.tWeights[prefix+'_BN_running_var']
        else:
            bn_mean = self._tVariables[prefix+'_BN_batch_mean']
            bn_var = self._tVariables[prefix+'_BN_batch_var']
        if self.params['static_mBN']:
            bn_mean = theano.gradient.disconnected_grad(bn_mean)
            bn_var = theano.gradient.disconnected_grad(bn_var)
        gamma = self.tWeights[prefix+'_W_BN_gamma']
        beta = self.tWeights[prefix+'_b_BN_beta']
        lin = T.dot(inp,W)+b
        bn_lin = gamma*(lin-bn_mean)/T.sqrt(bn_var+eps)+beta
        if onlyLinear:
            return bn_lin
        else:
            return self._applyNL(bn_lin,W)
        
    def _buildHiddenLayers(self, inp, nlayers, paramname, evaluation=False, normalization=True, modifiedBatchNorm=False):
        """
                 Convenience function to build hidden layers
        """
        for l in range(nlayers):
            myprefix = paramname.format(layer=l)
            Wname = myprefix+'_W'
            bname = myprefix+'_b'
            W=self.tWeights[Wname]
            bias=self.tWeights[bname]
            if self.params['batchnorm'] and normalization:
                if modifiedBatchNorm:
                    inp = self._BNlayerModified(W=W,b=bias,inp=inp,prefix=myprefix,evaluation=evaluation)
                else:
                    inp = self._BNlayer(W=W,b=bias,inp=inp,prefix=myprefix,evaluation=evaluation)
            elif self.params['layernorm'] and normalization:
                inp = self._LayerNorm(W=W,b=bias,inp=inp)
            else:
                inp = self._LinearNL(W=W,b=bias,inp=inp)
        return inp

    def _bilinear(self,x,W,y,b=None):
        """
                W should have shape (output_dim, x.shape[1], y.shape[1])
        """
        xW = T.dot(x,W)
        xWy = T.sum(xW*y.reshape((y.shape[0],1,-1)),axis=2)
        if b==None:
                return xWy
        else:
                return xWy+b

    def _logpdf_LogGamma(self, X, beta):
        """
                         log probability density function for loggamma
        """
        return (X*beta-T.exp(X)-T.gammaln(beta)).sum(axis=1,keepdims=True)

    def _LoggammaKL(self, beta, betaprior):
        """
                                KL Term for LogGamma Variates
        """
        KL = (T.gammaln(betaprior)-T.gammaln(beta)-(betaprior-beta)*Psi()(beta)).sum(1,keepdims=True)
        return KL
        
    
    def _variationalGaussian(self, mu, logcov, eps):
        """
                            KL divergence between N(0,I) and N(mu,exp(logcov))
        """
        #Pass z back
        z, KL = None, None
        if self.params['inference_model']=='single':
            z       = mu + T.exp(0.5*logcov)*eps
            KL      = 0.5*T.sum(-logcov -1 + T.exp(logcov) +mu**2 ,axis=1,keepdims=True)
        else:
            assert False,'Invalid inference model: '+self.params['inference_model']
        return z,KL

    def _variationalLoggamma(self, beta, betaprior):
        #generate loggamma variates, need to cut off gradient calcs through as_op
        #dim=[batchsize,nclasses]
        loggamma_variates = theano.gradient.disconnected_grad(self.rng_loggamma(beta))
        #calculate KL
        #dim=[batchsize,1]
        KL = self._LoggammaKL(beta, betaprior)
        return loggamma_variates, KL

    def _buildEmission(self, alpha, Z, X, graphprefix, evaluation=False, modifiedBatchNorm=False):
        """
                                Build subgraph to estimate conditional params
        """
        suffix = '_e' if evaluation else '_t'
        Z = self._buildHiddenLayers(Z,nlayers=self.params['z_generative_layers'],
                                    paramname='p_z_{layer}',
                                    evaluation=evaluation,
                                    normalization=self.params['p_normlayers'],
                                    modifiedBatchNorm=modifiedBatchNorm)
        if self.params['bilinear']:
            self._addVariable(graphprefix+'_p_Z_embedded'+suffix,Z,True)
            inp_p = self._bilinear(Z,self.tWeights['p_alpha_Z_input_W'],alpha)
            self._addVariable(graphprefix+'_p_alpha_Z_input_W'+suffix,inp_p,True)
            inp_p = T.nnet.softplus(inp_p)
        else:	
            inp_p = T.concatenate([alpha,Z],axis=1)
        inp_p = self._buildHiddenLayers(inp_p,nlayers=self.params['p_layers'],
                                        paramname='p_{layer}',
                                        evaluation=evaluation,
                                        normalization=self.params['p_normlayers'],
                                        modifiedBatchNorm=modifiedBatchNorm)

        if self.params['data_type']=='real':
            mu_p    = self._LinearNL(self.tWeights['p_mu_W'],self.tWeights['p_mu_b'],inp_p, onlyLinear=True)
            logcov_p= self._LinearNL(self.tWeights['p_logcov_W'],self.tWeights['p_logcov_b'],inp_p, onlyLinear=True)
            negCLL_m= 0.5 * (np.log(2 * np.pi) + logcov_p + ((X - mu_p) / T.exp(0.5*logcov_p))**2)
            return (mu_p, logcov_p), negCLL_m.sum(1,keepdims=True)
        else:
            mean_p = T.nnet.sigmoid(self._LinearNL(self.tWeights['p_mean_W'],self.tWeights['p_mean_b'],inp_p,onlyLinear=True))
            negCLL_m = T.nnet.binary_crossentropy(mean_p,X)
            return (mean_p,), negCLL_m.sum(1,keepdims=True)

    def _buildClassifier(self, logbeta, Y):
        probs = T.nnet.softmax(logbeta)
        #T.nnet.categorical_crossentropy returns a vector of length batch_size
        loss= T.nnet.categorical_crossentropy(probs,Y) 
        ncorrect = T.eq(T.argmax(probs,axis=1),Y).sum()
        return probs, loss, ncorrect

    def _buildRecognitionBase(self, X, dropout_prob=0.,evaluation=False,modifiedBatchNorm=False,graphprefix=None):
        if self.params['modifiedBatchNorm']:
            self._BNprefix=None
        elif self.params['separateBNrunningstats']:
            self._BNprefix=graphprefix
        else:
            self._BNprefix=None
        self._p(('Inference with dropout :%.4f')%(dropout_prob))
        inp = self._dropout(X,dropout_prob)
        hx = self._buildHiddenLayers(inp,self.params['q_layers'],'q_h(x)_{layer}',evaluation,modifiedBatchNorm=modifiedBatchNorm)
        logbeta = self._buildHiddenLayers(hx,self.params['alpha_inference_layers'],'q_logbeta_{layer}',evaluation,modifiedBatchNorm=modifiedBatchNorm)
        if evaluation==False and self.params['dropout_logbeta'] > 0:
            logbeta = self._dropout(logbeta,self.params['dropout_logbeta']) 
        logbeta = self._LinearNL(W=self.tWeights['q_logbeta_out_W'],b=self.tWeights['q_logbeta_out_b'],inp=logbeta,onlyLinear=True)
        logbeta = T.clip(logbeta,-20,20)
        return hx, logbeta

    def _build_qz(self,alpha,hx,evaluation,modifiedBatchNorm,graphprefix):
        if self.params['dropout_hx']>0 and evaluation==False:
            hx = self._dropout(hx,self.params['dropout_hx'])
        hz = self._buildHiddenLayers(hx,self.params['hz_inference_layers'],'q_hz_{layer}',evaluation,modifiedBatchNorm=modifiedBatchNorm)
        if self.params['separateBNrunningstats']:
            self._BNprefix=graphprefix
            modifiedBatchNorm=False
        else:
            self._BNprefix=None
        suffix = '_e' if evaluation else '_t'
        #merge h(x) and alpha
        if self.params['bilinear']:
            self._addVariable(graphprefix+'_hz'+suffix,hz,True)
            hz_W_alpha = self._bilinear(hz,self.tWeights['q_alpha_h(x)_W'],alpha)
            self._addVariable(graphprefix+'_hz_W_alpha'+suffix,hz_W_alpha,True)
            hz_alpha = T.nnet.softplus(hz_W_alpha)
        else:
            alpha_embed = self._LinearNL(W=self.tWeights['q_alpha_h(x)_W'],b=self.tWeights['q_alpha_h(x)_b'],inp=alpha,onlyLinear=True)
            self._addVariable(graphprefix+'_alpha_embed'+suffix,alpha_embed,True)
            hz_alpha = T.concatenate([alpha_embed,hz],axis=1) 
        #infer mu and logcov
        q_Z_hidden = self._buildHiddenLayers(hz_alpha,self.params['z_inference_layers'],'q_Z_{layer}',evaluation,modifiedBatchNorm=modifiedBatchNorm)
        mu      = self._LinearNL(self.tWeights['q_Z_mu_W'],self.tWeights['q_Z_mu_b'],q_Z_hidden,onlyLinear=True)
        logcov  = self._LinearNL(self.tWeights['q_Z_logcov_W'],self.tWeights['q_Z_logcov_b'],q_Z_hidden,onlyLinear=True)
        return mu, logcov

    def _variationalDirichlet(self,beta,betaprior):
        #U = loggamma variates
        U, KL_loggamma = self._variationalLoggamma(beta,betaprior)
        #convert to Dirichlet (with sharpening)
        if self.params['sharpening'] != 1:
            alpha = T.nnet.softmax(U*self.params['sharpening'])
        else:
            alpha = T.nnet.softmax(U)
        return alpha, KL_loggamma

    def _maxBetaDistance(self,logbeta,weight=1):
        sortedBeta = T.sort(logbeta,axis=1)
        distance = sortedBeta[:,-1]-sortedBeta[:,-2]
        return -weight*distance

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

        beta = T.exp(logbeta)
        if Y!=None:
            if self.params['logpxy_discrete']:
                nllY = theano.shared(-np.log(0.1))
                KL_loggamma = theano.shared(0.)
                alpha = Y
            else:
                beta += Y 
                beta_y = (beta*Y).sum(axis=1)
                nllY = Psi()(beta.sum(axis=1)) - Psi()(beta_y)
                #U = loggamma variates
                U, KL_loggamma = self._variationalLoggamma(beta,betaprior)
                #convert to Dirichlet (with sharpening)
                if self.params['sharpening'] != 1:
                    alpha = T.nnet.softmax(U*self.params['sharpening'])
                else:
                    alpha = T.nnet.softmax(U)
        else:
            #U = loggamma variates
            U, KL_loggamma = self._variationalLoggamma(beta,betaprior)
            #convert to Dirichlet (with sharpening)
            if self.params['sharpening'] != 1:
                alpha = T.nnet.softmax(U*self.params['sharpening'])
            else:
                alpha = T.nnet.softmax(U)

        #U = loggamma variates
        #U, KL_loggamma = self._variationalLoggamma(beta,betaprior)
        #convert to Dirichlet (with sharpening)
        #if self.params['sharpening'] != 1:
        #    alpha = T.nnet.softmax(U*self.params['sharpening'])
        #else:
        #    alpha = T.nnet.softmax(U)
        # q(z|x,alpha)
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
            KL = self.params['KL_loggamma_coef']*KL_loggamma.sum() + KL_Z.sum()
        else:
            KL = KL_loggamma.sum() + KL_Z.sum()
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

        #gradient hack to do black box variational inference:
        if evaluation == False:
            if Y==None or self.params['logpxy_discrete']==False:
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
            if Y==None:
                if self.params['maxBetaWeight'] != 0:
                    objfunc += self._maxBetaDistance(logbeta,self.params['maxBetaWeight']).sum()
            else:
                if self.params['maxBetaWeightXY'] != 0:
                    objfunc += self._maxBetaDistance(logbeta,self.params['maxBetaWeightXY']).sum()



        outputs = {'alpha':alpha,
                   'Z':Z,
                   'logbeta':logbeta,
                   'bound':bound,
                   'objfunc':objfunc,
                   'nllX':nllX,
                   'KL_loggamma':KL_loggamma,
                   'KL_Z':KL_Z,
                   'KL':KL,
                   'NLL':NLL}
        if Y!=None:
            outputs['nllY']=nllY

        #return alpha,Z,logbeta,bound,objfunc
        return outputs

    def _getModelOutputs(self,outputsU,outputsL,suffix=''):
        my_outputs = {
                         'KL_loggammaU':outputsU['KL_loggamma'].sum(),
                         'nllX_U':outputsU['nllX'].sum(),
                         'NLL_U':outputsU['NLL'].sum(),
                         'KL_U':outputsU['KL'].sum(),
                         'KL_Z_U':outputsU['KL_Z'].sum(),
                         'nllY':outputsL['nllY'].sum(),
                         'KL_loggammaL':outputsL['KL_loggamma'].sum(),
                         'nllX_L':outputsL['nllX'].sum(),
                         'NLL_L':outputsL['NLL'].sum(),
                         'KL_L':outputsL['KL'].sum(),
                         'KL_Z_L':outputsL['KL_Z'].sum(),
                         'logbeta_U':outputsU['logbeta'],
                         'logbeta_L':outputsL['logbeta'],
                         'mu_U':self._tVariables['U_mu'+suffix],
                         'mu_L':self._tVariables['L_mu'+suffix],
                         'logcov2_U':self._tVariables['U_logcov2'+suffix],
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
    
    def _buildModel(self):
        """
                                       ******BUILD DiscreteSemiVAE GRAPH******
        """
        #Inputs to graph
        XU   = T.matrix('XU',   dtype=config.floatX)
        XL   = T.matrix('XL',   dtype=config.floatX)
        Y   = T.ivector('Y')
        epsU = T.matrix('epsU', dtype=config.floatX)
        epsL = T.matrix('epsL', dtype=config.floatX)
        self._fakeData(XU,XL,Y,epsU,epsL)
        self.updates_ack = True
        #Learning Rates and annealing objective function
        #Add them to npWeights/tWeights to be tracked [do not have a prefix _W or _b so wont be diff.]
        self._addWeights('lr', np.asarray(self.params['lr'],dtype=config.floatX),borrow=False)
        self._addWeights('annealKL', np.asarray(0.,dtype=config.floatX),borrow=False)
        self._addWeights('annealCW', np.asarray(0.,dtype=config.floatX),borrow=False)
        self._addWeights('update_ctr', np.asarray(1.,dtype=config.floatX),borrow=False)
        
        lr  = self.tWeights['lr']
        annealKL = self.tWeights['annealKL']
        annealCW = self.tWeights['annealCW']
        iteration_t    = self.tWeights['update_ctr'] 
        
        lr_update = [(lr,T.switch(lr/1.0005<1e-4,lr,lr/1.0005))]
        annealKL_div     = float(self.params['annealKL']) #50000.
        annealCW_div     = float(self.params['annealCW']) #50000.
        ctr_update = [(iteration_t, iteration_t+1)]
        annealKL_update  = [(annealKL,T.switch(iteration_t/annealKL_div>1,1.,0.01+iteration_t/annealKL_div))]
        annealCW_update  = [(annealCW,T.switch(iteration_t/annealCW_div>1,1.,0.01+iteration_t/annealCW_div))]

        Y_onehot = T.extra_ops.to_one_hot(Y,self.params['nclasses'],dtype=config.floatX)
        meanAbsDev = 0
        #Build training graphs
        graphprefixU = 'U'
        graphprefixL = 'L'
        graphprefixC = 'q(y|x)'
        #_,_,_,boundU_t,objfuncU_t=self._buildGraph(XU,epsU,self.params['betaprior'],
        outputsU_t = self._buildGraph(XU,epsU,self.params['betaprior'],
                                      Y=None,
                                      dropout_prob=self.params['input_dropout'],
                                      add_noise=True,
                                      annealKL=annealKL,
                                      evaluation=False,
                                      modifiedBatchNorm=False,
                                      graphprefix=graphprefixU)#use BN stats from U for L
        boundU_t = outputsU_t['bound']
        objfuncU_t = outputsU_t['objfunc']
        outputsL_t = self._buildGraph(XL,epsL,self.params['betaprior'],
                                      Y=Y_onehot,
                                      dropout_prob=self.params['input_dropout'],
                                      add_noise=True,
                                      annealKL=annealKL,
                                      evaluation=False,
                                      modifiedBatchNorm=self.params['modifiedBatchNorm'],
                                      graphprefix=graphprefixL)
        logbetaL_t = outputsL_t['logbeta']
        boundL_t = outputsL_t['bound']
        objfuncL_t = outputsL_t['objfunc']
        if self.params['separateBNrunningstats']:
            _, logbeta_t = self._buildRecognitionBase(XL, dropout_prob=self.params['input_dropout'],evaluation=False,modifiedBatchNorm=self.params['modifiedBatchNorm'],graphprefix=graphprefixC)
            _,crossentropyloss_t,ncorrect_t = self._buildClassifier(logbeta_t,Y)
        else:
            _,crossentropyloss_t,ncorrect_t = self._buildClassifier(logbetaL_t,Y)
        trainboundU = boundU_t.sum()
        trainboundL = boundL_t.sum()
        trainclassifier = self.params['classifier_weight']*crossentropyloss_t.sum()
        trainbound = trainboundU + trainboundL 
        trainloss = trainbound + trainclassifier 
        trainobjective = objfuncU_t.sum() + objfuncL_t.sum() + annealCW*self.params['classifier_weight']*crossentropyloss_t.sum() 
        trainobj_components = [objfuncU_t.sum(),objfuncL_t.sum(),self.params['classifier_weight']*crossentropyloss_t.sum()]

        #Build evaluation graph
        outputsU_e = self._buildGraph(XU,epsU,self.params['betaprior'],
                                      Y=None,
                                      dropout_prob=0.,
                                      add_noise=False,
                                      annealKL=None,
                                      evaluation=True,
                                      graphprefix=graphprefixU)
        boundU_e = outputsU_e['bound']
        objfuncU_e = outputsU_e['objfunc']
        outputsL_e = self._buildGraph(XL,epsL,self.params['betaprior'],
                                      Y=Y_onehot,
                                      dropout_prob=0.,
                                      add_noise=False,
                                      annealKL=None,
                                      evaluation=True,
                                      modifiedBatchNorm=self.params['modifiedBatchNorm'],
                                      graphprefix=graphprefixL)
        logbetaL_e = outputsL_e['logbeta']
        boundL_e = outputsL_e['bound']
        objfuncL_e = outputsL_e['objfunc']
        if self.params['separateBNrunningstats']:
            _, logbeta_e = self._buildRecognitionBase(XL, dropout_prob=0.,evaluation=True,modifiedBatchNorm=self.params['modifiedBatchNorm'],graphprefix=graphprefixC)
            _,crossentropyloss_e,ncorrect_e = self._buildClassifier(logbeta_e,Y)
        else:
            _,crossentropyloss_e,ncorrect_e = self._buildClassifier(logbetaL_e,Y)
        evalboundU = boundU_e.sum()
        evalboundL = boundL_e.sum()
        evalbound = evalboundU + evalboundL
        evalclassifier = self.params['classifier_weight']*crossentropyloss_e.sum()
        evalloss = evalbound + evalclassifier
        evalobjective = objfuncU_e.sum() + objfuncL_e.sum() + self.params['classifier_weight']*crossentropyloss_e.sum()

        #Optimizer with specification for regularizer
        model_params = self._getModelParams()
        nparams = float(self._countParams(model_params))
        #setup grad norm (scale grad norm according to # parameters)
        if self.params['grad_norm'] == None:
            grad_norm_per_1000 = 1.0
        else:
            grad_norm_per_1000 = self.params['grad_norm']
        grad_norm = nparams/1000.0*grad_norm_per_1000
        self._p('# params to optimize = %s, max gradnorm = %s' % (nparams,grad_norm))

        if self.params['divide_grad']:
            divide_grad = T.cast(XU.shape[0],config.floatX)
        else:
            divide_grad = None
        optimizer_up, norm_list  = self._setupOptimizer(trainobjective, model_params,lr = lr, 
                                                        reg_type =self.params['reg_type'], 
                                                        reg_spec =self.params['reg_spec'], 
                                                        reg_value= self.params['reg_value'],
                                                       grad_norm = grad_norm, 
                                                       divide_grad = divide_grad) 
        #self.updates is container for all updates (e.g. see _BNlayer in __init__.py)
        self.updates += optimizer_up+annealKL_update+annealCW_update+ctr_update
        
        #Build theano functions
        fxn_inputs = [XU,XL,Y,epsU,epsL]
        
        #Importance sampled estimate
#        ll_prior             = self._llGaussian(z_e, T.zeros_like(z_e,dtype=config.floatX),
#                                                    T.zeros_like(z_e,dtype=config.floatX))
#        ll_posterior         = self._llGaussian(z_e, mu_e, logcov_e)
#        ll_estimate          = -1*negCLL_e+ll_prior.sum(1,keepdims=True)-ll_posterior.sum(1,keepdims=True)
#        self.likelihood      = theano.function(fxn_inputs,ll_estimate)
        #outputs_train = [trainloss,ncorrect_t,trainbound,anneal.sum(),trainboundU,trainboundL,trainclassifier]
        outputs_train = {'cost':trainloss,
                         'ncorrect':ncorrect_t,
                         'bound':trainbound,
                         'annealKL':annealKL.sum(),
                         'annealCW':annealCW.sum(),
                         'boundU':trainboundU,
                         'boundL':trainboundL,
                         'classification_loss':trainclassifier,
                         }
        for k,v in self._getModelOutputs(outputsU_t,outputsL_t,suffix='_t').iteritems():
            outputs_train[k] = v

        outputs_eval = { 'cost':evalloss,
                         'ncorrect':ncorrect_e,
                         'bound':evalbound,
                         'boundU':evalboundU,
                         'boundL':evalboundL,
                         'classification_loss':evalclassifier,
                        }
        for k,v in self._getModelOutputs(outputsU_e,outputsL_e,suffix='_e').iteritems():
            outputs_eval[k] = v

        # add batchnorm running statistics to output
        for k,v in self.tWeights.iteritems():
            if 'running' in k:
                outputs_train[k] = v

        self.train      = theano.function(fxn_inputs, outputs_train,  
                                              updates = self.updates, name = 'Train')
        outputs_debug = outputs_train #+ norm_list + trainobj_components
        self.debug      = theano.function(fxn_inputs, outputs_debug,
                                              updates = self.updates, name = 'Train+Debug')
#        self.inference  = theano.function(fxn_inputs, [z_e, mu_e, logcov_e], name = 'Inference')
        #outputs_eval = [evalloss,ncorrect_e,evalbound]
        self.evaluate   = theano.function(fxn_inputs, outputs_eval, name = 'Evaluate')
        self.decay_lr   = theano.function([],lr.sum(),name = 'Update LR',updates=lr_update)
#        self.reconstruct= theano.function([z_e], list(params_e), name='Reconstruct')
#        self.reset_anneal=theano.function([],anneal.sum(), updates = [(anneal,0.01)], name='reset anneal')
    def sample(self,nsamples=100):
        """
                                Sample from Generative Model
        """
        z = np.random.randn(nsamples,self.params['dim_stochastic']).astype(config.floatX)
        return self.reconstruct(z)
    
    def infer(self, data):
        """
                                Posterior Inference using recognition network
        """
        assert len(data.shape)==2,'Expecting 2D data matrix'
        assert data.shape[1]==self.params['dim_observations'],'Wrong dimensions for observations'
        
        eps  = np.random.randn(data.shape[0],self.params['dim_stochastic']).astype(config.floatX)
        return self.inference(X=data.astype(config.floatX),eps=eps)

    def evaluateBound(self, dataset, batch_size, S=1):
        """
                                    Evaluate bound S times on dataset 
        """
        N = dataset['X'].shape[0]
        outputs = {}
        #shuffle validation samples so that we grab a random output sample (for things
        #like logbetas...we don't want to save out all logbetas across entire validation
        #set)
        idxlist = np.arange(N)
        np.random.shuffle(idxlist)
        for bnum,st_idx in enumerate(range(0,N,batch_size)):
            end_idx = min(st_idx+batch_size, N)
            Nbatch = end_idx-st_idx
            idx = idxlist[st_idx:end_idx]
            minibatch = {'XU':self.sampleDataset(dataset['X'][idx]),
                         'XL':self.sampleDataset(dataset['X'][idx]),
                         'Y':dataset['Y'][idx].astype('int32')}
            for s in range(S):
                minibatch['epsU']=np.random.randn(Nbatch,self.params['dim_stochastic']).astype(config.floatX)
                minibatch['epsL']=np.random.randn(Nbatch,self.params['dim_stochastic']).astype(config.floatX)
                if self.params['inference_model']=='single':
                    #batch_cost,batch_ncorrect,batch_bound = self.evaluate(**minibatch)
                    batch_outputs = self.evaluate(**minibatch)
                    for k,v in batch_outputs.iteritems():
                        if k not in outputs.keys() or v.size>1:
                            outputs[k] = v 
                        else:
                            outputs[k] += v
                else:
                    assert False,'Should not be here'
        outputs['accuracy'] = outputs.pop('ncorrect')
        for k,v in sorted(outputs.iteritems(),key=lambda x:x[0]):
            if v.size <= 1:
                outputs[k] = v/float(N*S)
                print k, outputs[k]
        return outputs
    
    def meanSumExp(self,mat,axis=1):
        """
        Estimate log 1/S \sum_s exp[ log k ] in a numerically stable manner where axis represents the sum
        """
        a = np.max(mat, axis=1, keepdims=True)
        return a + np.log(np.mean(np.exp(mat-a.repeat(mat.shape[1],1)),axis=1,keepdims=True))
    
    def impSamplingNLL(self, dataset, batch_size, S = 200):
        """
                                    Importance sampling based log likelihood
        """
        N = dataset.shape[0]
        ll = 0
        for bnum,st_idx in enumerate(range(0,N,batch_size)):
            end_idx = min(st_idx+batch_size, N)
            X       = dataset[st_idx:end_idx].astype(config.floatX)
            
            batch_lllist = []
            for s in range(S):
                eps     = np.random.randn(X.shape[0],self.params['dim_stochastic']).astype(config.floatX)
                if self.params['inference_model']=='single':
                    batch_ll = self.likelihood(X=X, eps=eps)
                else:
                    assert False,'Should not be here'
                batch_lllist.append(batch_ll)
            ll  += self.meanSumExp(np.concatenate(batch_lllist,axis=1), axis=1).sum()
        ll /= float(N)
        return -ll

    def sampleDataset(self, dataset):
        p = np.random.uniform(low=0,high=1,size=dataset.shape)
        return (dataset >= p).astype(config.floatX)
    
    def _getMiniBatch(self,XU,XL,YL):
        return {'XU':self.sampleDataset(dataset['XU'][idxlist[st_idx:end_idx]]),
                'XL':self.sampleDataset(dataset['XL'][idx_labeled]),
                'Y':dataset['YL'][idx_labeled],
                'epsU':np.random.randn(Nbatch,self.params['dim_stochastic']).astype(config.floatX),
                'epsL':np.random.randn(Nbatch,self.params['dim_stochastic']).astype(config.floatX)}

    def learn(self, dataset, epoch_start=0, epoch_end=1000, batch_size=200, shuffle=True, 
              savedir = None, savefreq = None, evalfreq = 10, dataset_eval=None, replicate_K = None,
              debug=False):
        assert len(dataset['XU'].shape)==2,'Expecting 2D dataset matrix'
        assert dataset['XU'].shape[1]==self.params['dim_observations'],'dim observations incorrect'
        assert shuffle==True,'Shuffle should be true, especially when using batchnorm'
        N = dataset['XU'].shape[0]
        idxlist = range(N)
        Nlabeled = dataset['XL'].shape[0]
        cost_train,cost_valid,accuracy_train,accuracy_valid,trainbound,validbound=[],[],[],[],[],[]
        trainboundU,trainboundL,trainclassifier = [],[],[]
        trainBNrunningstats = {}
        train_logbetas = {} 
        train_hx = {}
        validBNrunningstats = {}
        valid_logbetas = {} 
        valid_hx = {}
        current_lr = self.params['lr']

        train_outputs = {}
        valid_outputs = {}
        track_params = {}

        not_normalized = ['annealKL','annealCW']
        #Training epochs
        for epoch in range(epoch_start, epoch_end+1):
            start_time = time.time()
            cost, bound, grad_norm, param_norm = 0, 0, [], []
            boundU,boundL,classifier = 0,0,0
            num_predictions, ncorrect = 0,0
            debug_outputs = []
            epoch_outputs = {}
            
            #Update learning rate
            if self.params['lr_decay']:
                current_lr = self.decay_lr()

            if shuffle:
                np.random.shuffle(idxlist)

            #Go through dataset
            for bnum,st_idx in enumerate(range(0,N,batch_size)):
                end_idx = min(st_idx+batch_size, N)
                Nbatch  = end_idx-st_idx
                idx_labeled = np.random.randint(low=0,high=Nlabeled,size=Nbatch)
                minibatch = {'XU':self.sampleDataset(dataset['XU'][idxlist[st_idx:end_idx]]),
                             'XL':self.sampleDataset(dataset['XL'][idx_labeled]),
                             'Y':dataset['YL'][idx_labeled],
                             'epsU':np.random.randn(Nbatch,self.params['dim_stochastic']).astype(config.floatX),
                             'epsL':np.random.randn(Nbatch,self.params['dim_stochastic']).astype(config.floatX)}
                num_predictions += minibatch['Y'].shape[0]
                
                #Forward/Backward pass
                if debug:
                    outputs = self.debug(**minibatch)
                    batch_cost,batch_ncorrect,batch_bound,anneal = outputs[:4]
                    debug_outputs.append(outputs[4:])
                else:
                    #batch_cost,batch_ncorrect,batch_bound,anneal,batch_boundU,batch_boundL,batch_classifier = self.train(**minibatch)
                    batch_outputs = self.train(**minibatch)
                    batch_cost = batch_outputs['cost']
                    batch_ncorrect = batch_outputs['ncorrect']
                    batch_bound = batch_outputs['bound']
                    annealKL = batch_outputs['annealKL']
                    annealCW = batch_outputs['annealCW']
                    batch_boundU = batch_outputs['boundU']
                    batch_boundL = batch_outputs['boundL']
                    batch_classifier = batch_outputs['classification_loss']
                    for k,v in batch_outputs.iteritems():
                        if k not in epoch_outputs or v.size > 1:
                            epoch_outputs[k] = v 
                        elif k in not_normalized:
                            epoch_outputs[k] = np.hstack([epoch_outputs[k],v])
                        else:
                            epoch_outputs[k] += v 
                
                #Divide value of the bound by replicateK
                #if replicate_K is not None:
                #    batch_cost /= float(replicate_K)
                #    batch_bound /= float(replicate_K)
                
 #               cost += batch_cost
 #               bound += batch_bound
 #               boundU += batch_boundU
 #               boundL += batch_boundL
 #               classifier += batch_classifier
 #               ncorrect += batch_ncorrect
                if bnum%100==0:
                    self._p(('--Batch: %d, Batch Loss: %.2f, Accuracy: %.3f, Anneal [KL,CW]: [%.2f,%0.2f] Lr: %.2e--')%
                            (bnum, batch_cost/Nbatch, batch_ncorrect/float(Nbatch), annealKL,annealCW, current_lr))
                
#            bound /= float(N)
#            cost /= float(N)
#            accuracy = ncorrect/float(num_predictions)
#            boundU /= float(N)
#            boundL /= float(N)
#            classifier /= float(N)

            epoch_outputs['accuracy'] = epoch_outputs.pop('ncorrect')

            for k,v in sorted(epoch_outputs.iteritems(),key=lambda x:x[0]):
                if v.size <= 1:
                    epoch_outputs[k] = v/float(N)
                    print k, epoch_outputs[k]
                if k not in train_outputs.keys():
                    train_outputs[k] = epoch_outputs[k]
                elif k in not_normalized:
                    train_outputs[k] = np.hstack([train_outputs[k],epoch_outputs[k]])
                else:
                    train_outputs[k] = np.vstack([train_outputs[k],epoch_outputs[k]])
            if 'epochs' not in train_outputs:
                train_outputs['epochs'] = np.array(epoch)
            else:
                train_outputs['epochs'] = np.hstack([train_outputs['epochs'],epoch]) 
#                if 'running' in k:
#                    if k in trainBNrunningstats.keys():
#                        trainBNrunningstats[k] = np.vstack([trainBNrunningstats[k],v])
#                    else:
#                        trainBNrunningstats[k] = v 
#                elif 'logbeta' in k:
#                    if k in train_logbetas.keys():
#                        train_logbetas[k] = np.vstack([train_logbetas[k],v])
#                    else:
#                        train_logbetas[k] = v
#                elif 'h(x)' in k:
#                    if k in train_hx.keys():
#                        train_hx[k] = np.vstack([train_hx[k],v])
#                    else:
#                        train_hx[k] = v
#                else:
#                    print k,v/float(N)

            

#            cost_train.append((epoch,cost))
#            accuracy_train.append((epoch,accuracy))
#            trainbound.append((epoch,bound))
#            trainboundU.append((epoch,boundU))
#            trainboundL.append((epoch,boundL))
#            trainclassifier.append((epoch,classifier))
#            end_time   = time.time()
#            if debug:
#                import cPickle as pickle
#                with open(os.path.join(savedir,'debug.pkl'),'w') as f:
#                    pickle.dump({'debug_outputs':debug_outputs,
#                                 'params':self.params,
#                                 'train_accuracy':accuracy_train,
#                                 'train_bound':trainbound,
#                                 'train_cost':cost_train},f)
            cost = epoch_outputs['cost']
            accuracy = epoch_outputs['accuracy']
            boundU = epoch_outputs['boundU']
            boundL = epoch_outputs['boundL']
            classifier = epoch_outputs['classification_loss']
            self._p(('Ep (%d) Train: Loss %.2f, Accuracy %0.3f, Bound (U=%.2f,L=%.2f), Classifier %0.2f [Took %.2f seconds]')%(epoch, cost, accuracy, boundU, boundL, classifier, time.time()-start_time))

            if epoch%20 == 0:
                print 'experiment: %s' % savedir
            if evalfreq is not None and epoch%evalfreq==0:
                #v_cost,v_accuracy,v_bound = self.evaluateBound(dataset_eval,batch_size,replicate_K)
                v_outputs = self.evaluateBound(dataset_eval,batch_size,replicate_K)
                v_cost = v_outputs['cost']
                v_accuracy = v_outputs['accuracy']
                v_bound = v_outputs['bound']
#                cost_valid.append((epoch,v_cost))
#                accuracy_valid.append((epoch,v_accuracy))
#                validbound.append((epoch,v_bound))
                for k,v in v_outputs.iteritems():
                    if k not in valid_outputs.keys():
                        valid_outputs[k] = v
                    else:
                        valid_outputs[k] = np.vstack([valid_outputs[k],v])
                if 'epochs' not in valid_outputs:
                    valid_outputs['epochs'] = np.array(epoch)
                else:
                    valid_outputs['epochs'] = np.hstack([valid_outputs['epochs'],epoch]) 
#                for k,v in v_outputs.iteritems():
#                    if 'running' in k:
#                        if k in validBNrunningstats.keys():
#                            validBNrunningstats[k] = np.vstack([validBNrunningstats[k],v])
#                        else:
#                            validBNrunningstats[k] = v 
#                    elif 'logbeta' in k:
#                        if k in valid_logbetas.keys():
#                            valid_logbetas[k] = np.vstack([valid_logbetas[k],v])
#                        else:
#                            valid_logbetas[k] = v
#                    elif 'hx' in k:
#                        if k in valid_hx.keys():
#                            valid_hx[k] = np.vstack([valid_hx[k],v])
#                        else:
#                            valid_hx[k] = v
#                v_ll = self.impSamplingNLL(dataset_eval, batch_size=batch_size)
#                validll.append((epoch,v_ll))
#                self._p(('Ep (%d): Valid Bound: %.4f, Valid LL: %.4f')%(epoch, v_bound, v_ll))
                self._p(('Ep (%d): Valid Loss: %0.4f, Valid Accuracy: %0.4f, Valid Bound: %0.4f'%(epoch,v_cost,v_accuracy,v_bound)))


                if self.params['track_params']:
                    track_params_list=['p_0_W','p_0_b']
                    if not self.params['bilinear']:
                        track_params_list += ['q_alpha_h(x)_W','q_alpha_h(x)_b','q_Z_0_W','q_Z_0_b']
                    else:
                        track_params_list += ['q_alpha_h(x)_W','q_alpha_h(x)_b']

                    for p in track_params_list:
                        weight = self.tWeights[p].get_value()
                        weight = weight.reshape([1]+list(weight.shape))
                        if p not in track_params.keys():
                            track_params[p] = weight 
                        else:
                            track_params[p] = np.vstack([track_params[p],weight])
                    if 'epochs' not in track_params:
                        track_params['epochs'] = np.array(epoch)
                    else:
                        track_params['epochs'] = np.hstack([track_params['epochs'],epoch]) 
            self._p(('Total epoch time: %0.4f' % (time.time()-start_time)))
            


            save_map = {'train':train_outputs,
                        'valid':valid_outputs}
            if self.params['track_params']:
                save_map['params'] = track_params
            print 'savedir: %s' % savedir
            if savefreq is not None and epoch%savefreq==0:
                self._p(('Saving at epoch %d'%epoch))
                if self.params['savemodel']:
                    self._saveModel(fname=os.path.join(savedir,'EP%s'%epoch))
#                intermediate = {}
#                intermediate['train_cost']  = np.array(cost_train)
#                intermediate['train_accuracy'] = np.array(accuracy_train)
#                intermediate['train_bound'] = np.array(trainbound)
#                intermediate['valid_cost']  = np.array(cost_valid)
#                intermediate['valid_accuracy'] = np.array(accuracy_valid)
#                intermediate['valid_bound'] = np.array(validbound)
#                intermediate['train_BNrunningstats'] = trainBNrunningstats
#                intermediate['train_logbetas'] = train_logbetas
#                intermediate['valid_BNrunningstats'] = validBNrunningstats
#                intermediate['valid_logbetas'] = valid_logbetas
#                intermediate['valid_ll']    = np.array(validll)
#                intermediate['samples']     = self.sample()
                try:
                    os.system('mkdir -p %s' % savedir)
                except:
                    pass
                saveHDF5(os.path.join(savedir,'EP%s-stats.h5'%epoch), save_map)
#        ret_map={}
#        ret_map['train_cost']  = np.array(cost_train)
#        ret_map['train_accuracy'] = np.array(accuracy_train)
#        ret_map['train_bound'] = np.array(trainbound)
#        ret_map['valid_cost']  = np.array(cost_valid)
#        ret_map['valid_accuracy'] = np.array(accuracy_valid)
#        ret_map['valid_bound'] = np.array(validbound)
#        ret_map['train_BNrunningstats'] = trainBNrunningstats
#        ret_map['train_logbetas'] = train_logbetas
#        ret_map['valid_BNrunningstats'] = validBNrunningstats
#        ret_map['valid_logbetas'] = valid_logbetas
#        ret_map['valid_ll']    = np.array(validll)
        return save_map
