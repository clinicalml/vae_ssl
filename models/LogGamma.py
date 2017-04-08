import time
import os
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
import ipdb
import scipy
from scipy.misc import logsumexp
from OutputLog import OutputLog

from . import AbstractSemiVAE

class LogGammaSemiVAE(AbstractSemiVAE):

    def _build_generative(self, alpha, Z):
        """
        Build subgraph to estimate conditional params
        """
		with self.namespaces('p(x|z,alpha'):

			# first transform Z into another embedding layer
			with self.namespaces('h(z)'):		
				Z = self._buildHiddenLayers(Z,diminput=self.params['dim_stochastic']
										     ,dimoutput=self.params['p_dim_hidden']
											 ,nlayers=self.params['z_generative_layers'])

			# combine alpha and Z
            h = T.concatenate([alpha,Z],axis=1)

			# hidden layers for p(x|z,alpha)
			with self.namespaces('h(h(z),alpha)'):		
				h = self._buildHiddenLayers(h,diminput=self.params['p_dim_hidden']+self.params['nclasses']
											 ,dimoutput=self.params['p_dim_hidden']
											 ,nlayers=self.params['p_layers'])

			# calculate emission probability parameters
			if self.params['data_type']=='real':
				with self.namespaces('gaussian'):		
					with self.namespaces('hidden'):
						mu = self._linear(h,diminput=self.params['p_dim_hidden']
										   ,dimoutput=self.params['dim_observations'])
						logcov2 = self._linear(h,diminput=self.params['p_dim_hidden']
												,dimoutput=self.params['dim_observations'])
					params = {'mu':mu,'logcov2':logcov2}
			else:
				with self.namespaces('bernoulli'):		
					with self.namespaces('hidden'):
						h = self._linear(h,diminput=self.params['p_dim_hidden']
									      ,dimoutput=self.params['dim_observations'])
						p = T.nnet.sigmoid(h)
					params = {'p':p}
			return params

    def _build_classifier(self, logbeta, Y):
        probs = T.nnet.softmax(logbeta)
        #T.nnet.categorical_crossentropy returns a vector of length batch_size
        loss= T.nnet.categorical_crossentropy(probs,Y) 
        ncorrect = T.eq(T.argmax(probs,axis=1),Y).sum()
        return probs, loss, ncorrect

    def _build_hx_logbeta(self, X) 
		"""
		return h(x), logbeta(h(x))
		"""
		if not self._evaluating:
			X = self._dropout(X,self.params['input_dropout'])
			self._p(('Inference with dropout :%.4f')%(self.params['input_dropout']))


		with self.namespaces('q_h(x)'):
			hx = self._buildHiddenLayers(X,diminput=self.params['dim_observations']
										  ,dimoutput=self.params['q_dim_hidden']
										  ,nlayers=self.params['q_layers'])

		with self.namespaces('q_logbeta_hidden'):
			h_logbeta = self._buildHiddenLayers(hx,diminput=self.params['q_dim_hidden']
												  ,dimoutput=self.params['q_dim_hidden']
												  ,nlayers=self.params['alpha_inference_layers'])

		if not self._evaluating:
            h_logbeta = self._dropout(h_logbeta,self.params['dropout_logbeta']) 

		with self.namespaces('q_logbeta'):
			logbeta = self._linear(h_logbeta,diminput=self.params['q_dim_hidden']
											,dimoutput=self.params['nclasses'])

		#clip to avoid nans
        logbeta = T.clip(logbeta,-5,5)

        return hx, logbeta

    def _build_inference(self,alpha,hx):
		"""
		return q(z|alpha,h(x))
		"""

		if not self._evaluating:
            hx = self._dropout(hx,self.params['dropout_hx'])

		with self.namespaces('hz(hx)'):
			hz = self._buildHiddenLayers(hx,diminput=self.params['q_dim_hidden']
										   ,dimoutput=self.params['q_dim_hidden']
										   ,nlayers=self.params['q_layers'])

		with self.namespaces('hz(alpha)'):
            alpha_embed = self._linear(alpha,diminput=self.params['nclasses']
											,dimoutput=self.params['q_dim_hidden'])
			
		# concatenate hz and alpha_embed
		hz_alpha = T.concatenate([alpha_embed,hz],axis=1) 

        # infer mu and logcov2 for q(z|alpha,x)
		with self.namespaces("q(z|alpha,x)"):
			q_Z_h = self._buildHiddenLayers(hz_alpha,diminput=2*self.params['q_dim_hidden']
													,dimoutput=self.params['q_dim_hidden']
													,nlayers=self.params['z_inference_layers'])

			diminput = self.params['q_dim_hidden']
			dimoutput = self.params['dim_stochastic']

			with self.namespaces('params'):
				mu = self._linear(q_Z_h,diminput=self.params['q_dim_hidden']
									   ,dimoutput=self.params['dim_stochastic'])
				logcov2 = self._linear(q_Z_h,diminput=self.params['q_dim_hidden']
			  							    ,dimoutput=self.params['dim_stochastic'])

        return mu, logcov2

    def _buildVAE(self, X, eps, betaprior=0.2, Y=None):
        """
		Build VAE subgraph to do inference and emissions 
		(if Y==None, build upper bound of -logp(x), else build upper bound of -logp(x,y)

		returns a bunch of VAE outputs
        """
        if Y is None:
            self._p(('Building graph for lower bound of logp(x)'))
        else:
            self._p(('Building graph for lower bound of logp(x,y)'))

		# from now on, under this with statement, we will always be working
		# with the same set of namespaces, so lets set them as defaults.
		# (this is actually already the default, but let's put it here anyway)
		#
		# tWeights: container for all theano shared variables
		# tOutputs: container for variables that will be outputs of our theano functions
		# tBatchnormStats: container for batchnorm running statistics
		#	* we have separate batchnorm running stats for each of p(x), p(x,y), q(y|x)
		with self.set_attr('default_namespaces',['tWeights','tOutputs','tBatchnormStats']):

			# build h(x) and logbeta
			hx, logbeta = self._build_hx_logbeta(X)

			beta = T.exp(logbeta)
			if Y is not None: 
				"""
				-logp(x,y)
				"""

				if self.params['logpxy_discrete']:
					# assume alpha = Y
					nllY = theano.shared(-np.log(0.1))
					KL_loggamma = theano.shared(0.)
					alpha = Y
				else:
					if self.params['learn_posterior']:
						with self.namespaces('q(alpha|y)'):			
							posterior = self._addWeights('posterior',np.asarray(1.))
							beta += Y*T.nnet.softplus(posterior) 
					else:
						beta += Y

					# select beta_y
					beta_y = (beta*Y).sum(axis=1)

					# calculate -logp(Y|alpha)
					nllY = Psi()(beta.sum(axis=1)) - Psi()(beta_y)

					# loggamma variates
					U, KL_loggamma = self._variationalLoggamma(beta,betaprior)

					# convert to Dirichlet (with sharpening)
					sharpening = self.tHyperparams['sharpening']
					alpha = T.nnet.softmax(U*sharpening)
			else:
				"""
				-logp(x)
				"""

				# loggamma variates
				U, KL_loggamma = self._variationalLoggamma(beta,betaprior)

				# convert to Dirichlet (with sharpening)
				alpha = T.nnet.softmax(U*self.tHyperparams['sharpening'])

			with self.namespaces('q(z|x,alpha)'):
				mu, logcov2 = self._build_inference(alpha,hx)

			# gaussian variates
			Z, KL_Z = self._variationalGaussian(mu,logcov2,eps)

			if not self._evaluating:
				# adding noise during training usually helps performance
				Z = Z + self.srng.normal(Z.shape,0,0.05,dtype=config.floatX)

			# generative model
			paramsX = self._build_generative(alpha, Z)
			if self.params['data_type']=='real':
				nllX = self._nll_gaussian(X,*paramsX).sum(axis=1)
			else:
				nllX = self._nll_bernoulli(X,*paramsX).sum(axis=1)

			# negative of the lower bound
			KL = KL_loggamma.sum() + KL_Z.sum()
			NLL = nllX.sum()
			if Y is not None:
				NLL += nllY.sum()
			bound = KL + NLL 

			# objective function
			if self._evaluating:
				objfunc = bound 
			else: 
				# annealing (training only)
				anneal = self.tHyperparams['annealing']

				# annealed objective function
				objfunc = anneal['KL_alpha']*KL_loggamma.sum() + anneal['KL_Z']*KL_Z.sum() + NLL

				# gradient hack to do black box variational inference:
				if Y is None or self.params['logpxy_discrete']==False:
					# previous if statement checks to see if we need to do inference over alpha
					# when self.params['logpxy_discrete']=True, we assume p(alpha|Y)=Y

					# make sure sizes are correct to prevent unintentional broadcasting
					KL_Z = KL_Z.reshape([-1])
					nllX = nllX.reshape([-1])

					if self.params['negKL']:
						# the negative KL trick is something we found by accident that
						# works well for when alpha is assumed to be loggamma or dirichlet
						#negative KL trick :(
						f = theano.gradient.disconnected_grad(-2.*KL_Z+nllX)
					else:
						f = theano.gradient.disconnected_grad(anneal['KL_Z']*KL_Z+nllX)

					# apply gradient hack to objective function
					BBVIgradientHack = f*self._logpdf_LogGamma(U,beta).reshape([-1])
					objfunc += BBVIgradientHack.sum()

			self.tOutputs.update({
									'alpha':alpha,
									'U':U,
									'Z':Z,
									#'paramsX':paramsX[0],
									'logbeta':logbeta,
									'bound':bound,
									'objfunc':objfunc,
									'nllX':nllX,
									'KL_loggamma':KL_loggamma,
									'KL_Z':KL_Z,
									'KL':KL,
									'NLL':NLL,
									'eps':eps,
								 })
			if Y is not None:
				self.tOutputs.update({
									'nllY':nllY
									})

			return self.tOutputs

	def _buildSemiVAE(self,XU,XL,YL,epsU,epsL,betaprior)
		"""
		U + L + classifier_weight*q(y|x)
		notes:
		* when training, components of the objective function are annealed
		* use separate batchnorm running stats for each of the subgraphs p(x), p(x,y), q(y|x);
		  though, this is not the correct way to use batchnorm, it is convenient for modeling
		  semi supervised VAEs
		"""

		# set tOutputs and tBatchnormStats namespaces to 'p(x)'
		with self.namespaces('p(x)',['tOutputs','tBatchnormStats']):
			outputsU = self._buildVAE(XU,epsU,betaprior,Y=None)
			boundU = outputsU['bound']
			objfuncU = outputsU['objfunc']

		YL_onehot = T.extra_ops.to_one_hot(YL,self.params['nclasses'],dtype=config.floatX)

		# set tOutputs and tBatchnormStats namespaces to 'p(x,y)'
		with self.namespaces('p(x,y)',['tOutputs','tBatchnormStats']):
			outputsL = self._buildVAE(XL,epsL,betaprior,Y=YL_onehot)
			boundL = outputsL['bound']
			objfuncL = outputsL['objfunc']

		# set tOutputs and tBatchnormStats namespaces to 'q(y|x)'
		with self.namespace('q(y|x)',['tOutputs','tBatchnormStats']):
			_, logbeta = self._build_hx_logbeta(XL)
			_, crossentropyloss, ncorrect = self._build_classifier(logbeta,YL)

		# calculate bound, loss, and theano-specific objective function (w/ gradient hacks)
		boundU = boundU.sum()
		boundL = boundL.sum()
		bound = boundU + boundL 
		classifier = self.params['classifier_weight']*crossentropyloss.sum()
		loss = bound + classifier 
		if self._evaluating:
			objective = loss 
		else:
			anneal = self.tHyperparams['annealing']
			# note that objfunc* contains the gradient hack, whereas bound* does not
			objective = anneal['bound']*(objfuncU.sum() + self.params['boundXY_weight']*objfuncL.sum()) + anneal['classifier']*self.params['classifier_weight']*crossentropyloss.sum() 

		self.tOutputs.update({
								'boundU':boundU,
								'boundL':boundL,
								'bound':bound,
								'classifier':classifier,
								'loss':loss,
								'objective':objective,
								'accuracy':ncorrect,
							})
		return self.tOutputs

	def _setupHyperparameters(self):
		# learning rate
		lr = self._addWeights('lr',np.asarray(self.params['lr']))
		# unlike all other updates, lr_update will be updated in a different theano function
        self.lr_update = [(lr,T.switch(lr/1.0005<1e-4,lr,lr/1.0005))]

		# iteration
		t = self._addWeights('update_ctr',np.asarray(1.)
        self._addUpdate(t,t+1)

		with self.namespaces('annealing'):
			# annealing parameters
			aKL_Z = self._addWeights('KL_Z',np.asarray(0.))
			aKL_alpha = self._addWeights('KL_alpha',np.asarray(0.))
			aCW = self._addWeights('classifier',np.asarray(0.))
			aBP = self._addWeights('betaprior',np.asarray(0.))
			aBound = self._addWeights('bound',np.asarray(0.))
			aSH = self._addWeights('sharpening',np.asarray(0.))

			# divisors for updates
			aKL_Z_div = float(self.params['annealKL_Z']) #50000.
			aKL_alpha_div = float(self.params['annealKL_alpha']) #50000.
			aCW_div = float(self.params['annealCW']) #50000.
			aBP_div = float(self.params['annealBP']) #50000.
			aBound_div = float(self.params['annealBound']) #50000.
			aSH_div = float(self.params['annealSharpening'])

			# updates
			self._addUpdate(aKL_Z,T.switch(t/aKL_Z_div>1,1.,0.01+t/aKL_Z_div))
			self._addUpdate(aKL_alpha,T.switch(t/aKL_Z_div>1,1.,0.01+t/aKL_Z_div))
			self._addUpdate(aCW,T.switch(t/aCW_div>1,1.,0.01+t/aCW_div))
			self._addUpdate(aBP,T.switch(t/aBP_div>1,1.,0.01+t/aBP_div))
			self._addUpdate(aBound,T.switch(t/aBound_div>1,1.,0.01+t/aBound_div))
			self._addUpdate(aSH,T.switch(t/aSH_div>1,1.,0.01+t/aSH_div))

		# betaprior is annealed from self.params['betaprior'] to self.params['finalbeta']
        finalbeta  = float(self.params['finalbeta'])
        betaprior = self.params['betaprior']*(1-aBP)+self.params['finalbeta']*aBP

		# sharpening
		sharpening = self._addWeights('sharpening',np.asarray(self.params['sharpening'])),
        self._addUpdate(sharpening,self.params['sharpening']*0.5*(1.+aSH))

		# save all hyperparameters to tOutputs for access later
		self.tOutputs.update(self.tWeights)
		self.tOutputs['betaprior'] = betaprior
    
    def _buildModel(self):
        """
        Build training, evaluation, inference, and sampling graphs for SemiVAE
        """

        #Inputs to graph
        XU = T.matrix('XU',   dtype=config.floatX)
        XL = T.matrix('XL',   dtype=config.floatX)
        YL = T.ivector('YL')
        epsU = T.matrix('epsU', dtype=config.floatX)
        epsL = T.matrix('epsL', dtype=config.floatX)
        self._fakeData(XU,XL,YL,epsU,epsL)


		# set tWeights and tOutputs to hyperparameters
		with self.namespaces('hyperparameters',['tWeights','tOutputs']):
			self._setupHyperparameters()
			self.tHyperparams = self.tOutputs
		betaprior = self.tHyperparams['betaprior']

		# We will have a separate sub-namespace in tWeights for storing batchnorm statistics.
		# This is for two reasons:
		#  1) everything under self.tWeights['weights'] will be optimized against the training
		#     objective function, and so we want to keep the batchnorm stats out of there
		#  2) we will want to have separate batchnorm running statistics for each of the
	    #     subgraphs p(x), p(x,y), and q(y|x); this is do to modeling convenience.
		with self.namespace('batchnorm_statistics','tWeights')
			self.tBatchnormStats = self.tWeights

		# set the namespace for tWeights to 'weights'
		with self.namespace('weights','tWeights'):
			"""
			Build training graph
			"""
			# set the namespace for tOutputs to 'train'
			# note that we would not want to set tWeights to 'train',
			# because the training and evaluating graphs share weights 
			with self.namespace('train','tOutputs'):
				train_outputs = self._buildSemiVAE(XU,XL,YL,epsU,epsL,betaprior)
			
			"""
			Build evaluation graph
			"""
			# set the namespace for tOutputs to 'evaluate'
			# note that we would not want to set tWeights to 'evaluate',
			# because the training and evaluating graphs share weights 
			with self.namespace('evaluate','tOutputs'):
				# set the entire class in evaluate mode 
				# i.e. this sets self._evaluating=True (instead of being False)
				with self._evaluate():
					self._buildSemiVAE(XU,XL,YL,epsU,epsL,betaprior)
		
			"""
			Build sampling graph
			"""
			with self.namespace('samples','tOutputs'):
				with self._evaluate():
					Z = T.matrix('Z',dtype=config.floatX)
					alpha = T.matrix('alpha',dtype=config.floatX)
					with self.namespaces('p(x)',['tOutputs','tBatchnormStats']):
						self.tOutputs['probs'] = self._build_generative(alpha,Z)
					with self.namespace('p(x,y)',['tOutputs','tBatchnormStats']):
						self.tOutputs['probs'] = self._build_generative(alpha,Z)
				

		#Training objective
		trainobjective = self.tOutputs['train']['objective'] 

        #Optimize all weights in self.tWeights['weights']
		#Note:
		# * tWeights is a Namespace, a nested dictionary object with some helpful functions
		# * The .leaves() function returns all the leaves of a Namespace object
		with self.namespace('weights','tWeights')
			optimized_weights = self.tWeights.leaves()

        optimization_outputs = self._setupOptimizer(trainobjective,optimized_weights,
													lr=self.tHyperparams['lr'],
													optimizer=self.params['optimizer'], 
													reg_type =self.params['reg_type'], 
													reg_value= self.params['reg_value'],
													grad_norm=self.params['grad_norm'],
													divide_grad=self.params['divide_grad']
													)
        optimizer_up, norm_list, trainobjective = optimization_outputs 
        
        #save some optimization outputs
		with.self.namespace('train','tOutputs'):
			self.tOutputs.update({'pnorm':norm_list[0],
								  'gnorm':norm_list[1],
								  'objective':trainobjective})

        #self.updates is container for all updates (e.g. see self._addUpdates in AbstraceSemiVAE)
        self.updates += optimizer_up
        
        #Build theano functions
        fn_inputs = [XU,XL,YL,epsU,epsL]

		with self.namespace('train','tOutputs'):
			# .flatten(join='/') converts hierarchical Namespace to Namespace of depth 1
			# where keypaths are converted to string and joined by '/'
			# e.g. {'a':{'b':1}} becomes {'a/b':1}
        	self.train = theano.function(fn_inputs,self.tOutputs.flatten(join='/')
										,name='Train'
                                        ,updates=self.updates)

		with self.namespace('evaluate','tOutputs'):
			self.evaluate = theano.function(fn_inputs,self.tOutputs.flatten(join='/')
										   ,name='Evaluate')

		# This should be called only once per epoch
        self.decay_lr = theano.function([],self.tHyperparameters['lr'].sum()
									   ,name='Update LR'
									   ,updates=self.lr_update)

		with self.namespace('samples','tOutputs'):
			with self.namespace('p(x)','tOutputs'):
				self.sample_px = theano.function([Z,alpha],{'probs':self.tOutputs['probs']}
												,name='sample p(x)')
			with self.namespace('p(x,y)','tOutputs'):
				self.sample_pxy = theano.function([Z,alpha],{'probs':self.tOutputs['probs']}
							 					 ,name='sample p(x)')

    def sample_model(self,nsamples=100):
        """
                                Sample from Generative Model
        """
        K = self.params['nclasses']
		z = np.random.randn(K*nsamples,self.params['dim_stochastic']).astype(config.floatX)
		alpha = np.repeat((np.arange(K).reshape(1,-1) == np.arange(K).reshape(-1,1)).astype('int'),axis=0,repeats=nsamples).astype(config.floatX)
		return {'U':self.sample_px(z,alpha),'L':self.sample_pxy(z,alpha)}

