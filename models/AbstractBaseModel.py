from __future__ import division
__all__=['BaseModel','static','temporal']

import six.moves.cPickle as pickle
from collections import OrderedDict
import sys, time, os
import numpy as np
import gzip, warnings, theano
from theano import config
theano.config.compute_test_value = 'warn'
from theano.printing import pydotprint
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from ..utils.optimizer import adam,rmsprop
from theano.tensor.nnet.bn import batch_normalization

class BaseModel:
    """
    Base Level Class for ML Models
    -Base level class that keeps the bare minimum amount of functionality
    -Allows loading/saving model from checkpoints (including optimization parameters)
    
    TODO: Does preserve randomness, i.e the random seeds would be different when restarted (low priority)
    """
    def __init__(self, params, paramFile=None, reloadFile=None, 
            additional_attrs = {},
            dataset_train = np.array([0]), 
            dataset_valid= np.array([0]), 
            dataset_test = np.array([0]), 
            dataset = np.array([0])):
        """
        MLModel
        params : Hashtable with parameters relevant to the model at hand
        paramFile : Location to save parameter file
        reloadFile: [Reloading Model from npz]
        dataset: (Optional) This is used when the dataset needs to be on the GPU. 
                            Set dataset to be the training data. During learning, you can then index 
                            self.dataset (which will be a theano shared variable) like so:
                            X = self.dataset_train[bidx] to represent the current batch during training
                            (similarly X = self.dataset_eval[bidx])
                            This will save on I/O but will cost you space on the GPU
        """
        np.random.seed(params['seed'])
        self.dataset_train = theano.shared(dataset_train.astype(config.floatX))
        self.dataset_valid = theano.shared(dataset_valid.astype(config.floatX))
        self.dataset_test  = theano.shared(dataset_test.astype(config.floatX))
        self.Ntrain        = dataset_train.shape[0]
        self.Nvalid        = dataset_valid.shape[0]
        self.Ntest         = dataset_test.shape[0]
        self.dataset       = theano.shared(dataset.astype(config.floatX))
        self.Ndataset      = dataset.shape[0]
        assert paramFile is not None,'Need to specify paramFile, either to create or to load from'
        if reloadFile is not None:
            self._p('Reloading Model')
            self._loadModel(reloadFile, paramFile)
        else:
            self.params     = params
            self.npWeights = self._createParams()
            assert self.npWeights is not None,'Expecting self.npWeights to be defined' 
            self._saveParams(paramFile)
        self.tWeights = self._makeTheanoShared(self.npWeights)
        #Added on May 10: Backwards compatibility. If you add more parameters to params, you'd like to 
        #be able to add them into the params so that you can build old models on potentially different
        #configurations. 
        for k in params:
            if (k in params and k not in self.params) or self.params[k]!=params[k]:
                print 'Adding/Modifying loaded parameters: ',k,' to ',params[k]
                self.params[k]= params[k]
        self._countParams()
        if hasattr(self, 'npOptWeights'):
            self.tOptWeights  = self._makeTheanoShared(self.npOptWeights)
        else:
            self.npOptWeights = None
            self.tOptWeights  = None
        start_time = time.time()
        self.srng = RandomStreams(params['seed'])
        #Use this for updates that you might need to specify differently from your training function. 
        #self.updates is a container for all theano function updates
        self.updates = []

        #Adding additional attributes to the base class
        for attr in additional_attrs:
            val = additional_attrs[attr]
            if val is not None and val.__class__.__name__=='ndarray':
                self._p('Setting '+attr+' as theano shared variable')
                setattr(self,attr,theano.shared(val.astype(config.floatX)))
            else:
                self._p('Setting '+attr+' to '+str(val))
                setattr(self,attr,val)

        self._buildModel()
        self._p(('_buildModel took : %.4f seconds')%(time.time()-start_time))
        assert self.tOptWeights is not None, 'Need to have optimization weights specified when building model'

    def _p(self,stringToPrint,logThis=False):
        """
        _p: print formatted string
        """
        toPrint = '\t<<'+str(stringToPrint)+'>>'
        print toPrint
        if logThis and hasattr(self,'logf'):
            self.logf.write(toPrint+'\n')

    def _openLogFile():
        assert 'logfile' in self.params,'Requires location of logfile'
        self.logf = open(self.params['logfile'],'a')
    def _closeLogFile():
        self.logf.close()
    """
    Saving and loading Model
    """
    def _loadModel(self, reloadFile, paramFile):
        """
        _loadModel: paramFile contains the model structure and reloadFile contains the weights
        (The optimization file is inferred from reloadFile)
        """
        assert os.path.exists(reloadFile),'Checkpoint file not found: '+reloadFile
        assert os.path.exists(paramFile),'Paramfile not found: '+paramFile
        optFile = reloadFile.replace('-params','-optParams')
        assert os.path.exists(optFile),'Optfile not found: '+optFile
        self._p(('Loading structure (%s) and model (%s) / opt (%s) weights')%(paramFile,reloadFile,optFile))
        self.params        = self._loadPickle(paramFile)
        self.npWeights     = np.load(reloadFile)
        self.npOptWeights  = np.load(optFile)
        
    def _saveModel(self,fname = None):
        """
        _saveModel: Save model to "fname". Uses a separate file for parameters and optimization params
        """
        assert fname is not None, 'Specify a save file'
        fname_par = os.path.splitext(fname)[0]+'-params'
        fname_opt = os.path.splitext(fname)[0]+'-optParams'
        weights_par = self._getValuesFromShared(self.tWeights)
        np.savez(fname_par, **weights_par)
        weights_opt = self._getValuesFromShared(self.tOptWeights)
        np.savez(fname_opt, **weights_opt)
        self._p(('Saved model (%s) \n\t\t opt (%s) weights')%(fname_par,fname_opt))
    
    def _getValuesFromShared(self,dictIn):
        """
        _getValuesFromShared: Get numpy arrays from theano shared variables in dictIn
        """
        new_params = OrderedDict()
        for kk, vv in dictIn.items():
            new_params[kk] = vv.get_value()
        return new_params
    
    def _countParams(self):
        """
        _countParams: Count the number of parameters in the model
        """
        self.nParams    = 0
        for k in self.npWeights:
            ctr = np.array(self.npWeights[k].shape).prod()
            self.nParams+= ctr
        self._p(('Nparameters: %d')%self.nParams)
    
    def _loadPickle(self,f):
        """
        _loadPickle: Load (first item) from pickle file
        """
        with open(f,'rb') as pklf:
            data = pickle.load(pklf)
        return data
    def _savePickle(self,f,data):
        """
        _savePickle: Save data to pickle file
        """
        with open(f,'wb') as pklf:
            pickle.dump(data,pklf) 
    def _saveParams(self, pklname = None):
        """
        _saveParams: Save data to pickle file
        """
        assert pklname is not None,'Expecting name of file to be saved'
        self._savePickle(pklname, self.params)
    
    def _addWeights(self, name, data, **kwargs):
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
            warnings.warn(name+" found in npWeights. No action taken")

    def _addUpdate(self, var, data):
        """
        Add an update for tWeights
        """
        if len(self.updates) > 0 and var in zip(*self.updates)[0]:
            warnings.warn(var.name+' found in self.updates...no action taken')
        else:
            self.updates.append((var,data))


    def _getModelParams(self, restrict = ''):
        """
        Return list of model parameters to take derivatives with respect to
        """
        paramlist = []
        namelist  = []
        for k in self.tWeights.values():
            if 'W_' in k.name or 'b_' in k.name or '_b' in k.name or '_W' in k.name or 'U_' in k.name or '_U' in k.name:
                #Use this to only get a list of parameters with specific substrings like 'p_'
                #Since it is set to '' by default, it should appear in all strings
                if restrict in k.name:
                    paramlist.append(k)
                    namelist.append(k.name)
        self._p('Modifying : ['+','.join(namelist)+']')
        return paramlist
    
    def _makeTheanoShared(self, dictIn):
        """
        _makeTheanoShared:  return an Ordered dictionary with the same keys as "dictIn" 
                        except with elements initialized to theano shared variables
        """
        tWeights = OrderedDict()
        for kk, pp in dictIn.items():
            tWeights[kk] = theano.shared(dictIn[kk], name=kk, borrow=True)
        return tWeights
    
    
    def _checkMatrix(self, mat):
        """
        Use to debug functions. Check if any element is nan or inf and print the norm
        """
        if np.any(np.isnan(mat)):
            self._p('checkMatrix: NaN found')
        if np.any(np.isinf(mat)):
            self._p('checkMatrix: inf found')
        self._p('Norm: %.4f:'%np.linalg.norm(mat))
    
    """
    Initializing weights of model    
    """
    def _getWeight(self, shape, scheme = None):
        """
        _getWeight: Wrapper for initializing weights
        Assumes w = self.params['init_weight'] has been set
        
        lstm: Initializing LSTM weights using orthogonal weight matrices
        and large forget gate biases
        
        uniform: 
        """
        if scheme is None: #Default to the pre-specified scheme
            scheme = self.params['init_scheme']
        assert 'init_weight' in self.params,'Error:Init. weight not specified in params.'
        if scheme=='lstm':
            return self._getLSTMWeight(shape)
        elif scheme=='orthogonal':
            return self._getOrthogonalWeight(shape)
        elif scheme=='gmm_mu' or scheme=='gmm_logcov':
            return self._gmmWeight(shape, scheme)
        elif scheme == 'uniform':
            return self._getUniformWeight(shape)
        elif scheme == 'normal':
            return self._getGaussianWeight(shape)
        elif scheme == 'xavier':
            return self._getXavierWeight(shape)
        elif scheme == 'he':
            return self._getHe2015(shape)
        else:
            return self._getUniformWeight(shape)
    
    def _getLSTMWeight(self, shape):
        """
        http://yyue.blogspot.com/2015/01/a-brief-overview-of-deep-learning.html
        For LSTMs, use orthogonal initializations for the weight matrices and 
        set the forget gate biases to be high
        """
        if len(shape)==1: #bias
            dim = int(shape[0]/4)
            self._p('Sampling biases for LSTM from exponential distribution')
            return np.random.laplace(size=shape).astype(config.floatX)
            #return np.concatenate([self._getUniformWeight((dim,)),np.ones((dim,))*self.params['forget_bias'],
            #                       self._getUniformWeight((dim*2,))]).astype(config.floatX)
        elif len(shape)==2: #weight
            nin = shape[0]
            nout= shape[1]
            assert int(nout/4)==nin,'Not LSTM weight.'
            return np.concatenate([self._getOrthogonalWeight((nin,int(nout/4))),
                                   self._getOrthogonalWeight((nin,int(nout/4))),
                                   self._getOrthogonalWeight((nin,int(nout/4))),
                                   self._getOrthogonalWeight((nin,int(nout/4)))]
                                  ,axis=1).astype(config.floatX)
        else:
            assert False,'Should not get here'
        
    def _getUniformWeight(self, shape):
        """
        _getUniformWeight: Initialize weight matrix of dimensions "shape" using uniform 
                    [-self.params['init_weight'], self.params['init_weight']]
        """
        return np.random.uniform(-self.params['init_weight'],self.params['init_weight'],shape).astype(config.floatX)
    
    def _getGaussianWeight(self, shape):
        """
        Initialize weight matrix of dimensions "shape" using normal with variance
                    given by self.params['init_weight']
        """
        return np.random.normal(0,self.params['init_weight'],shape).astype(config.floatX)

    def _getHe2015(self, shape):
        #http://cs231n.github.io/neural-networks-2/
        if len(shape)==1:
            return np.random.normal(0,self.params['init_weight'],shape).astype(config.floatX)
        initializer = 'uniform'
        if self.params['nonlinearity']=='relu':
            K = np.sqrt(2./float((1+self.params['leaky_param']**2)*(shape[0])))
        else:
            K = np.sqrt(1./float(shape[0]))
    
        if initializer=='uniform':
            return np.random.uniform(-K,K,shape).astype(config.floatX)
        elif initializer=='normal':
            return np.random.normal(0,K,shape).astype(config.floatX)
        else:
            assert False,'Invalid initializer in _getXavierWeight'
            
    def _getXavierWeight(self, shape):
        """
        Xavier Initialization
        """
        #Initialize biases randomly
        if len(shape)==1:
            return np.random.normal(0,self.params['init_weight'],shape).astype(config.floatX)
        initializer = 'uniform'
        if self.params['nonlinearity'] =='relu':
            K    = np.sqrt(12/float(shape[0]+shape[1]))
        else:
            K    = np.sqrt(2/float(shape[0]+shape[1]))
        if initializer=='uniform':
            return np.random.uniform(-K,K,shape).astype(config.floatX)
        elif initializer=='normal':
            return np.random.normal(0,K,shape).astype(config.floatX)
        else:
            assert False,'Invalid initializer in _getXavierWeight'
    
    def _getOrthogonalWeight(self, shape):
        """
        _getWeight: Initialize weight matrix of dimensions "shape" with orthonal columns
        """
        if len(shape)==1 or shape[0]!=shape[1]:
            self._p('shape not square, falling back to uniformly sampled weights')
            return self._getUniformWeight(shape)
        assert type(shape),'Expecting tuple in shape'
        W    = np.random.randn(*shape)
        q, r = np.linalg.qr(W)
        return q.astype(config.floatX)
    
    def _applyNL(self,lin_out):
        if self.params['nonlinearity']=='relu':
            if 'leaky_params' in self.params:
                return T.nnet.relu(lin_out, alpha = self.params['leaky_params'])
            else:
                return T.nnet.relu(lin_out)
        elif self.params['nonlinearity']=='softplus':
            return T.nnet.softplus(lin_out)
        elif self.params['nonlinearity']=='elu':
            return T.switch(lin_out > 0, lin_out, T.exp(lin_out) - 1)
        elif self.params['nonlinearity']=='maxout':
            maxout_out = None
            for i in xrange(self.params['maxout_stride']):
                tmp = lin_out[:,i::self.params['maxout_stride']]
                if maxout_out is None:
                    maxout_out = tmp
                else:
                    maxout_out = T.maximum(maxout_out, tmp)
            return maxout_out
        else:
            return T.tanh(lin_out)
    
    def _BNlayer(self, W, b, inp, onlyLinear = False, evaluation=False, convolution=False,momentum=0.95,eps=1e-3):
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
        #gamma_init = self._getWeight((W_shape[1],))
        #beta_init = self._getWeight((W_shape[1],))
        gamma_init = np.ones((W_shape[1],),dtype=config.floatX)
        beta_init  = np.zeros((W_shape[1],),dtype=config.floatX)
        gamma_name = W_name+'_BN_gamma'
        beta_name  = W_name+'_BN_beta'
        self._addWeights(gamma_name, gamma_init, borrow=True)
        self._addWeights(beta_name,  beta_init,  borrow=True)
        #Create a running mean that will not be differentiated 
        mean_name  = W_name.replace('W','').replace('__','_')+'BN_running_mean'
        var_name   = W_name.replace('W','').replace('__','_')+'BN_running_var'
        mean_init  = np.zeros((W_shape[1],), dtype=config.floatX)
        var_init   = np.ones((W_shape[1],),  dtype=config.floatX)
        self._addWeights(mean_name, mean_init, borrow=True)
        self._addWeights(var_name,  var_init, borrow=True)
        #momentum set to 0 in first iteration
        self._addWeights('BN_momentum', np.asarray(0.,dtype=config.floatX), borrow=True)
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
                normalized     = (lin-cur_mean) / T.sqrt(cur_var+eps)
                bn_lin         = self.tWeights[gamma_name]*normalized + self.tWeights[beta_name]
                #Update running stats
                self._addUpdate(self.tWeights[mean_name], self.tWeights['BN_momentum'] * self.tWeights[mean_name] + (1.0-self.tWeights['BN_momentum']) * cur_mean)
                self._addUpdate(self.tWeights[var_name],  self.tWeights['BN_momentum'] * self.tWeights[var_name] + (1.0-self.tWeights['BN_momentum']) * (float(W_shape[0])/float(W_shape[0]-1))* cur_var)
                #momentum will be 0 in the first iteration
                self._addUpdate(self.tWeights['BN_momentum'],momentum)
        #Elementwise nonlinearity
        lin_out = bn_lin
        if onlyLinear:
            return lin_out
        else:
            return self._applyNL(lin_out)

    def _LayerNorm(self, W, b, inp, onlyLinear=False, eps=1e-6):
        """
                               Layer Normalization 
        Implementation of https://arxiv.org/abs/1607.06450
        a) First perform linear transformation T.dot(inp,W)+b
        b) W must have at least 2 dimensions
        c) Create gain & bias, create updates for gain & bias
        """
        W_name     = W.name
        W_shape    = self.npWeights[W_name].shape
        ndims = len(W_shape)
        assert ndims>=2,'Expecting W to have at least 2 dimensions'
        gain_init = np.ones(tuple(W_shape[1:]),dtype=config.floatX)
        bias_init  = np.zeros(tuple(W_shape[1:]),dtype=config.floatX)
        gain_name = W_name+'_LayerNorm_gain'
        bias_name  = W_name+'_LayerNorm_bias'
        self._addWeights(gain_name, gain_init, borrow=True)
        self._addWeights(bias_name,  bias_init,  borrow=True)
        gain = self.tWeights[gain_name]
        bias = self.tWeights[bias_name]
        #layer norm transformation
        lin = T.dot(inp,W)+b
        mean = lin.mean(tuple(range(1,ndims)),keepdims=True)
        var = lin.var(tuple(range(1,ndims)),keepdims=True)
        normalized = (lin-mean) / T.sqrt(var+eps)
        LN_output = gain*normalized + bias
        if onlyLinear:
            return LN_output 
        else:
            return self._applyNL(LN_output)

    def _LinearNL(self, W, b, inp, onlyLinear=False):
        """
        _LinearNL : if onlyLinear: return T.dot(inp,W)+b else return NL(T.dot(inp,W)+b)
        """
        lin = T.dot(inp,W)+b
        lin_out = lin
        #If only doing a dot product return as is
        if onlyLinear:
            return lin_out
        else:
            return self._applyNL(lin_out)
    def _LinearDropoutNL(self, W, b, inp, p=0.):
        """ Linear + Dropout+ NL """
        lin = T.dot(inp,W)+b
        dlin= self._dropout(lin, p=p)
        return self._applyNL(dlin)
    
    def _dropout(self, X, p=0.):
        """
        _dropout : X is the input, p is the dropout probability
        Do not need to do anything in the case of no dropout since we divide by retain prob.
        """
        if p > 0:
            retain_prob = 1 - p
            X *= self.srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
            X /= retain_prob
        return X
    
    def _setupOptimizer(self, cost, params, lr, **kwargs):
        """
        _setupOptimizer :   Wrapper for calling optimizer specified for the model. Internally also updates
                            the list of shared optimization variables in the model
        Calls self.optimizer to minimize "cost" wrt "params" using learning rate "lr", the other arguments are passed
        as is to the optimizer

        returns: updates (list of tuples specifying updates for all the shared variables in the model)
                 norm_list (for debugging, [0] : norm of parameters, [1] : norm of gradients, [2] : norm of optimization weights)
        """
        optimizer_up, norm_list, opt_params = self.optimizer(cost, params, lr=lr, 
                                                             opt_params = self.tOptWeights, **kwargs)
        #If we passed in None initially then set optWeights
        if self.tOptWeights is None:
            self.tOptWeights = opt_params
        return optimizer_up, norm_list
    
    
    def _llGaussian(self, z, mu, logcov, mix_probs = None):
        """
        Estimate log-likelihood under a gaussian distribution
        """
        return -0.5*(np.log(2*np.pi)+logcov+((z-mu)**2/T.exp(logcov)))

    """
                                 Implementation of LSTMs
    """
    def _LSTMlayer(self, inp, suffix, dropout_prob=0., RNN_SIZE = None):
        """
        LSTM layer that takes as input inp [bs x T x dim] and returns the result of running an LSTM on it
        Input: inp [bs x T x dim]
               suffix [l/r]
               dropout applied at output of LSTM
        Output of LSTM:hid [T  x bsx dim]
 
        This function expects the following to be defined:
        params: rnn_size, nonlinearity, rnn_layers
        tWeights: W_lstm_<suffix>_0, U_lstm_<suffix>_0, b_lstm_<suffix>_0
            and if rnn_layers==2 we require W_lstm_<suffix>_1, U_lstm_<suffix>_1, b_lstm_<suffix>_1
        """
        self._p(('In _LSTMlayer with dropout %.4f')%(dropout_prob))
        if RNN_SIZE is None:
            RNN_SIZE = self.params['rnn_size']
        #Add support for bidirectional RNN
        assert suffix=='r' or suffix=='l' or suffix=='p_l','Invalid suffix: '+suffix
        doBackwards = False
        if suffix=='r':
            doBackwards = True 
        #Get Slice
        def _slice(mat, n, dim):
            if mat.ndim == 3:
                return mat[:, :, n * dim:(n + 1) * dim]
            return mat[:, n * dim:(n + 1) * dim]
        ###### LSTM
        def _1layer(x_,  h_, c_, lstm_U):
            preact = T.dot(h_, lstm_U)
            preact += x_
            i = T.nnet.sigmoid(_slice(preact, 0, RNN_SIZE))
            f = T.nnet.sigmoid(_slice(preact, 1, RNN_SIZE))
            o = T.nnet.sigmoid(_slice(preact, 2, RNN_SIZE))
            c = T.tanh(_slice(preact, 3, RNN_SIZE))
            c = f * c_ + i * c
            h = o * T.tanh(c)
            return h, c
        assert self.params['rnn_layers']==1,'Only 1/2 layer LSTM supported'
        inp_swapped= inp.swapaxes(0,1)
        #Perform the single matrix multiply for all the inputs across time
        lstm_embed = T.dot(inp_swapped,self.tWeights['W_lstm_'+suffix+'_0'])+ self.tWeights['b_lstm_'+suffix+'_0']
        nsteps     = lstm_embed.shape[0]
        n_samples  = lstm_embed.shape[1]
        stepfxn    = _1layer
        o_info     =[T.alloc(np.asarray(0.,dtype=config.floatX), n_samples, RNN_SIZE),
                T.alloc(np.asarray(1.,dtype=config.floatX), n_samples, RNN_SIZE) ]
        n_seq      =[self.tWeights['U_lstm_'+suffix+'_0']]

        lstm_input = lstm_embed
        #Reverse the input
        if doBackwards:
            lstm_input = lstm_input[::-1]
        rval, _= theano.scan(stepfxn, 
                              sequences=[lstm_input],
                              outputs_info=o_info,
                              non_sequences = n_seq,
                              name='LSTM_'+suffix, 
                              n_steps=nsteps)
        #set the output
        lstm_output =  rval[0]
        #Reverse the output
        if doBackwards: 
            lstm_output = lstm_output[::-1]
        return self._dropout(lstm_output, dropout_prob)
    
    def meanSumExp(self,mat,axis=1):
        """
        Estimate log 1/S \sum_s exp[ log k ] in a numerically stable manner along "axis"
        """
        a = np.max(mat, axis=axis, keepdims=True)
        return a + np.log(np.mean(np.exp(mat-a.repeat(mat.shape[axis],axis)),axis=axis,keepdims=True))

    def logsumexp(self, mat, axis=None):
        mat_max = T.max(mat, axis=axis, keepdims=True)
        lse = T.log(T.sum(T.exp(mat - mat_max), axis=axis, keepdims=True)) + mat_max
        return lse

