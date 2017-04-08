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
from . import BaseModel
import ipdb
import random
import scipy
from scipy.misc import logsumexp
from contextlib import contextmanager
from contextlib import nested 
from OutputLog import OutputLog
from Namespace import Namespace

IGNORE_WARNINGS=True

class Weight(object):

class AbstractSemiVAE(BaseModel, object):
    def __init__(self, params, paramFile=None, reloadFile=None):
		self._evaluating = False
        super(AbstractSemiVAE,self).__init__(params, paramFile=paramFile, reloadFile=reloadFile)

	def getNamespace(self,attr_name):
        if not hasattr(self,'__namespace_paths__'):
            #__namespace_paths__ keeps track of the path through the nested dictionary
            self.__namespace_paths__ = {}
        if attr_name not in self.__namespace_paths__:
            self.__namespace_paths__[attr_name] = []
		if not hasattr(self,attr_name):
			attr = Namespace()
		else:
			attr = getattr(self,attr_name)
			assert isinstance(attr,dict), '%s must be a type of dict' % attr_name

	@contextmanager
	def namespace(self,key,attr_name='tWeights'):
		"""
		* sets self.__dict__[attr_name] to self.__dict__[attr_name][key]
		* if attr_name does not exist, then it is created
		* if key does not exist in attr_name, then it is created
		example usage:
		```
		self.tWeights['layer_weights'] = {'w':np.random.randn(5)}
		with self.attr_name('layer_weights','tWeights'):
			w = self.tWeights['w']

		self.tWeights['layers'] = {'layer1':{'w':np.random.randn(5)}}
		with self.attr_name('layers','tWeights'), self.attr_name('layer1','tWeights'):
			w = self.tWeights['w']
		```
		"""
		#do this stuff before executing stuff under the with statement:
        if key not in attr:
            attr[key] = Namespace()
        temp = attr
        setattr(self,attr_name,attr[key])
        self.__namespace_paths__[attr_name].append(key)
		yield #wait until we finish executing the with statement"
		#now that we've exited with statement, do this stuff:
        setattr(self,attr_name,temp)
        self.__namespace_paths__[attr_name].pop(key)

	default_namespaces = ['tWeights','tOutputs']

    def namespace_path(self,attr_name):
        if hasattr(self,'__namespace_paths__') and attr_name in self.__namespace_paths__:
            return self.__namespace_paths__[attr_name]
        else:
            return []


	@contextmanager
	def namespaces(self,keys,attr_names=None):
		"""
		converts attr_names and keys to lists, if they are not already lists
		for a in attr_names:
			for k in keys:
				set the self.namespace(a,k)

		if attr_names is None, then attr_names=self.default_namespaces

		example usage:
		```
		self.attr1 = {}
		self.attr2 = {}
		with self.namespaces(['key1','key2'],['attr1','attr2']):
			self.attr1['A'] = 1
			self.attr2['B'] = 2
		print self.attr1
		>> {'key1':{'key2':{'A':1}}}
		print self.attr2
		>> {'key1':{'key2':{'B':2}}}
		```
		"""
		if attr_names is None:
			attr_names = self.default_namespaces
        if not isinstance(attr_names,list):
            attr_names=[attr_names]
        if not isinstance(keys,list):
            keys =[keys]
        assert len(attr_names) > 0, 'len(attr_names) cannot be zero'
        assert len(keys) > 0, 'len(keys) cannot be zero'
        
        managers = []
        for a in attr_names:
            for k in keys:
                managers.append(self.namespace(k,a))
        with nested(*managers):
            yield

	@contextmanager
	def _evaluate(self):
		"""
		sets self._evaluating=True

		example usage:
		```
		with self._evaluate():
			#do stuff
		```
		"""
		if not hasattr(self,'_evaluating'):
			self._evaluating=False
		self._evaluating=True
		yield
		self._evaluating=False

	@contextmanager
	def set_attr(self,attr_name,value):
		"""
		sets attr_name to value
		"""
		if not hasattr(self,attr_name):
			setattr(self,attr_name,None)
		temp = getattr(self,attr_name)
		setattr(self,attr_name,value)
		yield
		setattr(self,attr_name,temp)

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
			

    def _addShared(self,name,data,namespace_name,ignore_warnings=IGNORE_WARNINGS,**kwargs):
        """
        Add theano shared to namespace 
        
        name in theano.shared will be '/'.join(self.namespace_path('tWeights')+[name])
        e.g. if current namespace is tWeights['weights']['layer1'] and weight has name 'W'
        the fullpath will be 'weights/layer1/W'
        """
		namespace = getattr(self,namespace_name)
        if name not in namespace:
			fullpath = '/'.join([str(s) for s in namespace.path()]+[name])
            namespace[name]  = theano.shared(data.astype(config.floatX),name=fullpath,**kwargs)
        else:
            if not ignore_warnings:
                warnings.warn(name+" found in tWeights. No action taken")
		return namespace[name]

    def _addWeights(self, name, data, **kwargs):
        """
        Add to tWeights (under current namespace)
        
        name in theano.shared will be '/'.join(self.namespace_path('tWeights')+[name])
        e.g. if current namespace is tWeights['weights']['layer1'] and weight has name 'W'
        the fullpath will be 'weights/layer1/W'
        """
		return self._addShared(name,data,'tWeights',**kwargs)

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
    
    def batchnorm(self, x, dimoutput, output_axis, ndims, bias=True, momentum=0.95, eps=1e-3, **kwargs):
        """
		Batch normalization layer
		https://arxiv.org/abs/1502.03167
		* x: input to batch normalization
		* dimoutput: size of output_axis dimensions
		* output_axis: a separate set of batchnorm stats will be generated for each element in output_axis (e.g. following linear layers, output_axis=1; following convolutional layers, output_axis should be the channels dimension)
        """
		bn_shape = (dimoutput,)
		gamma = self._addWeights('bn_gamma',self._getWeight(bn_shape,**kwargs))

		
		running_mean = self._addShared('bn_running_mean',np.zeros(bn_shape),'tBatchnormStats')
		running_var = self._addShared('bn_running_var',np.ones(bn_shape),'tBatchnormStats')
		mom = self._addShared('bn_momentum',np.asarray(0),'tBatchnormStats')

		if self._evaluating:
			y = (x-running_mean)/T.sqrt(running_var+eps)
		else:
			if not hasattr(output_axis,'__iter__'):
				output_axis = [output_axis]
			#set of axes we will calculate batch norm statistics over
			axis = [for i in range(ndims) if i not in output_axis]
			batch_mean = x.mean(axis)
			batch_var = x.var(axis)
			y = (x-batch_mean)/T.sqrt(batch_var+eps)
			
			#Update running stats
			m = T.cast(x.shape[0],config.floatX)
			self._addUpdate(running_mean, mom*running_mean+(1.-mom)*batch_mean)
			self._addUpdate(running_var, mom*running_var+(1.-mom)*batch_var*m/(m-1))
			#momentum will be 0 in the first iteration, and momentum in all subsequent iters
			self._addUpdate(mom,momentum)

		z = gamma*normalized
		if bias:
			beta = self._addWeights('bn_beta',self._getWeight(shape,**kwargs))
			z = z+beta

		return z
        
    def _linear(self, x, diminput, dimoutput, bias=True, **kwargs):
        """
        * return T.dot(x,W)+b 
		* set bias=False to remove bias
        """
		W = self._addWeights('W',self._getWeight((diminput,dimoutput),**kwargs)
        y = T.dot(inp,W)
		if bias:
			b = self._addWeights('b',self._getWeight((dimoutput,))
			y = y + b
	
        #If only doing a dot product return as is
		return y

    def _LinearNL(self,*args,**kwargs) 
        """
        _LinearNL : if onlyLinear: return T.dot(x,W)+b else return NL(T.dot(inp,W)+b)
		* set bias=False to remove bias
        """
		y = self._linear(*args,**kwargs)
		return self._applyNL(y)
        
    def _bilinear(self,x,y,dimx,dimy,dimoutput,bias=True,**kwargs):
        """
		return xTWy+bW should have shape (output_dim, x.shape[1], y.shape[1])
        """
		W = self._addWeights('W',self._getWeight((dimoutput,dimx,dimy),**kwargs)
        xW = T.dot(x,W)
        xWy = T.sum(xW*y.reshape((y.shape[0],1,-1)),axis=2)
        if bias:
			b = self._addWeights('b',self._getWeight((dimoutput,),**kwargs))
			xWy = xWy+b
		return xWy

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
		z = mu + T.exp(0.5*logcov)*eps
		KL = 0.5*T.sum(-logcov -1 + T.exp(logcov) +mu**2 ,axis=1,keepdims=True)
        return z,KL

    def _variationalLoggamma(self, beta, betaprior):
        #generate loggamma variates, need to cut off gradient calcs through as_op
        #dim=[batchsize,nclasses]
        loggamma_variates = theano.gradient.disconnected_grad(self.rng_loggamma(beta))
        #calculate KL
        #dim=[batchsize,1]
        KL = self._LoggammaKL(beta, betaprior)
        return loggamma_variates, KL

    def _variationalDirichlet(self,beta,betaprior):
        #U = loggamma variates
        U, KL_loggamma = self._variationalLoggamma(beta,betaprior)
        #convert to Dirichlet (with sharpening)
        if self.tWeights['sharpening'] != 1:
            alpha = T.nnet.softmax(U*self.tWeights['sharpening'])
        else:
            alpha = T.nnet.softmax(U)
        return alpha, KL_loggamma

	def _nll_gaussian(self,X,mu,logcov2):
		return 0.5*(np.log(2*np.pi)+logcov2+((X-mu)/T.exp(0.5*logcov2))**2)

	def _nll_bernoulli(self,X,p):
		return T.nnet.binary_crossentropy(p,X)
		

    def _setupOptimizer(self,objective,namespace,lr,optimizer='adam',reg_value=0,reg_type='l2',divide_grad=True,grad_norm=None,**kwargs):
        """
        _setupOptimizer :   Wrapper for calling optimizer specified for the model. Internally also updates
                            the list of shared optimization variables in the model
        Calls self.optimizer to minimize "cost" wrt "params" using learning rate "lr", the other arguments are passed
        as is to the optimizer

        returns: updates (list of tuples specifying updates for all the shared variables in the model)
                 norm_list (for debugging, [0] : norm of parameters, [1] : norm of gradients, [2] : norm of optimization weights)
        """
		if self.params['optimizer']=='adam':
            optimizer = adam
        elif self.params['optimizer']=='rmsprop':
            optimizer = rmsprop
        else:
            assert False,'Invalid optimizer'

		# namespace.leaves() will yield a depth first iterator of namespace
		weights = namespace.leaves()

		#count number of weights for gradient normalization
        nparams = float(self._countParams(weights))

		#Add regularization
		if reg_value > 0:
            objective = optimize.regularize(objective,weights,reg_value,reg_type)

		#gradients
		grads = T.grad(objective,weights)

		#normalize gradient according to mini-batch size
        if divide_grad:
            divide_grad = T.cast(XU.shape[0],config.floatX)
			grads = optimize.rescale(grads,divide_grad)

        #setup grad norm (scale grad norm according to # parameters)
        if grad_norm is not None:
            grad_norm_per_1000 = self.params['grad_norm']
			grad_norm = nparams/1000.0*grad_norm_per_1000
			grads = optimize.normalize(grads, grad_norm)

        self._p('# params to optimize = %s, max gradnorm = %s' % (nparams,grad_norm))

        outputs = optimizer(params,grads,lr=lr,opt_params=self.tOptWeights,**kwargs)
        optimizer_up, norm_list, opt_params = outputs
 
        #If we passed in None initially then set optWeights
        if self.tOptWeights is None:
            self.tOptWeights = opt_params
        return optimizer_up, norm_list, objective

    def _buildHiddenLayers(self, h, diminput, dimoutput, nlayers, **kwargs)
        """
        Convenience function to build hidden layers
        """
		if self.params['nonlinearity']=='maxout':
			window = self.params['maxout_stride']
		else:
			window = 1
        for l in range(nlayers):
			with self.namespaces('layer'+str(l)):
				h = self._linear(h,diminput,window*dimoutput,**kwargs)
				if self.params['batchnorm'] and normalization:
                    inp = self.batchnorm(h,window*dimoutput,output_axis=1,ndims=2,**kwargs)
				elif self.params['layernorm'] and normalization:
					inp = self._LayerNorm(W=W,b=bias,inp=inp)
				h = self._applyNL(h)
				diminput = dimoutput
        return h

    def sample_dataset(self, dataset):
        p = np.random.uniform(low=0,high=1,size=dataset.shape)
        return (dataset >= p).astype(config.floatX)
    
    def preprocess_minibatch(self,minibatch):
		nU = minibatch['XU'].shape[0]
		nL = minibatch['XL'].shape[0]
        return {
				'XU':self.sample_dataset(minibatch['XU']),
				'XL':self.sample_dataset(minibatch['XL']),
				'YL':minibatch['YL'],
                'epsU':np.random.randn(nU,self.params['dim_stochastic']).astype(config.floatX),
                'epsL':np.random.randn(nL,self.params['dim_stochastic']).astype(config.floatX)
			   }

	def run_epoch(self,dataset,runfunc,maxiters=None,collect_garbage=False):
		start_time = time.time()
		epoch_outputs = OutputLog(axis=0,expand_dim=None)
		nbatches = len(dataset)
		with ProgressBar(nbatches) as pb:
			for i,data in enumerate(dataset):
				if collect_garbage:
					gc.collect()
				if maxiters is not None and i >= maxiters:
					break
				minibatch = self.preprocess_minibatch(data)
				# minibatch is assumed to be a dict
				batch_outputs = runfunc(**minibatch)
				epoch_outputs.add(batch_outputs)
				pb.update(i+1,self.progressBarUpdate(epoch_outputs))
		duration = time.time() - start_time
		epoch_outputs.add({'duration (seconds)':duration})
		return epoch_outputs

    def progressBarUpdate(self,epoch_outputs={}):
		# use list to preserve order
		report_map = [
			('boundU',np.mean,'%0.2f (epoch mean)'),
			('boundL',np.mean,'%0.2f (epoch mean)'),
			('bound',np.mean,'%0.2f (epoch mean)'),
			('classifier',np.mean,'%0.2f (epoch mean)'),
			('loss',np.mean,'%02.f (epoch mean)'),
			('objective',np.mean,'%02.f (epoch mean)'),
			('accuracy',np.mean,'%02.f (epoch mean)'),
			('hyperparameters/betaprior',lambda x:np.mean(x[-1]),'%05.f (last)'),
			('hyperparameters/lr',lambda x:np.mean(x[-1]),'%02.e (last)'),
			('hyperparameters/annealing/annealCW',lambda x:np.mean(x[-1]),'%02.e (last)'),
			('hyperparameters/annealing/annealKL_Z',lambda x:np.mean(x[-1]),'%02.e (last)'),
			('hyperparameters/annealing/annealKL_alpha',lambda x:np.mean(x[-1]),'%02.e (last)'),
		]
		report = []
		for k,f,s in report_map:
			if k in epoch_outputs:
				report.append(s.format(f(epoch_outputs[k])))
		if len(report)>0:
			return '\n' + '\n'.join(report)
		else:
			return None


    def learn(self, dataset, epoch_start=0, epoch_end=1000, batchsize=200, maxiters=None 
              savedir=None, savefreq=None, evalfreq=None, predfreq=None): 

        traindata = DataLoader(dataset.train,batchsize,shuffle=True)
        validdata = DataLoader(dataset.valid,batchsize,shuffle=False)

		log = OutputLog({'train':{},'valid':{}})	
		log_verbose = OutputLog({'train':{},'valid':{}})	
        log_samples = OutputLog({'samples':{}}

        for epoch in range(epoch_start,epoch_end+1):
            #train
            epoch_log = self.run_epoch(traindata,self.train,maxiters)
			log['train'].add(epoch_log.apply(np.mean))
			log['train'].add({'epoch':epoch})
			log_verbose['train'].add(epoch_log)

            if evalfreq is not None and epoch % evalfreq==0:
                #evaluate
				epoch_log = self.run_epoch(validdata,self.evaluate,maxiters)
				log['valid'].add(epoch_log.apply(np.mean))
				log['valid'].add({'epoch':epoch})
				log_verbose['valid'].add(epoch_log)

				#generate samples
				log_samples.add(self.sample_model(nsamples=100))

            if savefreq is not None and epoch % savefreq==0:
                self._p(('Saving at epoch %d'%epoch))
                self._p(('savedir: %s' % savedir))
                if self.params['savemodel']:
                    self._saveModel(fname=savedir)
                try:
                    os.system('mkdir -p %s' % savedir)
                except:
                    pass
                saveHDF5(os.path.join(savedir,'output.h5'), log)
                saveHDF5(os.path.join(savedir,'output_verbose.h5'), log_verbose)
                saveHDF5(os.path.join(savedir,'samples.h5'), log_samples)
            
        return log
            

