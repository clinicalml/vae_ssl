from AbstractModel import * 

class AbstractSemiVAE(AbstractModel):

    def __init__(self,params,*args,**kwargs):
        params['data_type'] = 'binary'
        super(AbstractSemiVAE,self).__init__(params,*args,**kwargs)

    def sample_dataset(self, dataset):
        p = np.random.uniform(low=0,high=1,size=dataset.shape)
        return (dataset >= p).astype(config.floatX)

    def preprocess_minibatch(self,minibatch):
        nU = len(minibatch['U']['X'])
        nL = len(minibatch['L']['X'])
        return {
                'XU':self.sample_dataset(minibatch['U']['X']),
                'XL':self.sample_dataset(minibatch['L']['X']),
                'YL':minibatch['L']['Y'].astype('int32'),
                'epsU':np.random.randn(nU,self.params['dim_stochastic']).astype(config.floatX),
                'epsL':np.random.randn(nL,self.params['dim_stochastic']).astype(config.floatX)
               }

    def progress_bar_report_map(self):
        # see self.progress_bar_update for use
        # use list to preserve order
        return [
            ('accuracy',lambda x:x.astype(float).mean(),'%0.2f (epoch mean)'),
            ('loss',np.mean,'%0.2f (epoch mean)'),
            #('objective',np.mean,'%0.2f (epoch mean)'),
            ('classifier',np.mean,'%0.2f (epoch mean)'),
            ('bound',np.mean,'%0.2f (epoch mean)'),
            ('boundU',np.mean,'%0.2f (epoch mean)'),
            ('boundL',np.mean,'%0.2f (epoch mean)'),
            ('hyperparameters/lr',lambda x:np.mean(x[-1]),'%0.2e (last)'),
            ('hyperparameters/annealing/classifier',lambda x:np.mean(x[-1]),'%0.2e (last)'),
            ('hyperparameters/annealing/KL_Z',lambda x:np.mean(x[-1]),'%0.2e (last)'),
            ('hyperparameters/annealing/KL_alpha',lambda x:np.mean(x[-1]),'%0.2e (last)'),
        ]

    def build_classifier(self, XL, Y):
        """
        calculate -log(q(Y|XL))

        return probs, loss, accuracy
        """
        pass

    def build_semi_vae(self,XU,XL,YL,epsU,epsL):
        """
        U + L + classifier_weight*q(y|x)
        notes:
        * when training, components of the objective function are annealed
        * use separate batchnorm running stats for each of the subgraphs p(x), p(x,y), q(y|x);
          though, this is not the correct way to use batchnorm, it is convenient for modeling
          semi supervised VAEs
        """

        # from now on, under this with statement, we will almost always be working
        # with the same set of namespaces, unless otherwise stated, so lets set them as defaults.
        # (this is actually already the default, but let's put it here anyway)
        #
        # tWeights: container for all theano shared variables
        # tOutputs: container for variables that will be outputs of our theano functions
        # tBatchnormStats: container for batchnorm running statistics
        # note: we have separate batchnorm running stats for each of p(x), p(x,y), q(y|x)
        with self.set_attr('default_namespaces',['tWeights','tOutputs','tBatchnormStats']):

            # set tOutputs and tBatchnormStats namespaces to 'p(x)'
            with self.namespaces('p(x)',['tOutputs','tBatchnormStats']):
                outputsU = self.build_vae(XU,epsU,Y=None)
                boundU = outputsU['bound']
                objfuncU = outputsU['objfunc']

            YL_onehot = T.extra_ops.to_one_hot(YL,self.params['nclasses'],dtype=config.floatX)

            # set tOutputs and tBatchnormStats namespaces to 'p(x,y)'
            with self.namespaces('p(x,y)',['tOutputs','tBatchnormStats']):
                outputsL = self.build_vae(XL,epsL,Y=YL_onehot)
                boundL = outputsL['bound']
                objfuncL = outputsL['objfunc']

            # set tOutputs and tBatchnormStats namespaces to 'q(y|x)'
            with self.namespaces('q(y|x)',['tOutputs','tBatchnormStats']):
                _, crossentropyloss, accuracy = self.build_classifier(XL,YL)

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

        # for reporting purposes, normalize some of the outputs
        self.tOutputs.update({
                                'boundU':boundU/XU.shape[0],
                                'boundL':boundL/XU.shape[0],
                                'bound':bound/XU.shape[0],
                                'classifier':classifier/XL.shape[0],
                                'loss':loss/XU.shape[0],
                                'objective':objective,
                                'accuracy':accuracy,
                            })
        return self.tOutputs

    def build_hyperparameters(self):
        # learning rate
        lr = self.add_weights('lr',np.asarray(self.params['lr']))
        # unlike all other updates, lr_update will be updated in a different theano function
        self.lr_update = [(lr,T.switch(lr/1.0005<1e-4,lr,lr/1.0005))]

        # iteration
        t = self.add_weights('update_ctr',np.asarray(1.))
        self.add_update(t,t+1)

        with self.namespaces('annealing'):
            # annealing parameters
            aKL_Z = self.add_weights('KL_Z',np.asarray(0.))
            aKL_alpha = self.add_weights('KL_alpha',np.asarray(0.))
            aCW = self.add_weights('classifier',np.asarray(0.))
            aBound = self.add_weights('bound',np.asarray(0.))
            aSH = self.add_weights('sharpening',np.asarray(0.))

            # divisors for updates
            aKL_Z_div = float(self.params['annealKL_Z']) #50000.
            aKL_alpha_div = float(self.params['annealKL_alpha']) #50000.
            aCW_div = float(self.params['annealCW']) #50000.
            aBound_div = float(self.params['annealBound']) #50000.
            aSH_div = float(self.params['annealSharpening'])

            # updates
            self.add_update(aKL_Z,T.switch(t/aKL_Z_div>1,1.,0.01+t/aKL_Z_div))
            self.add_update(aKL_alpha,T.switch(t/aKL_Z_div>1,1.,0.01+t/aKL_Z_div))
            self.add_update(aCW,T.switch(t/aCW_div>1,1.,0.01+t/aCW_div))
            self.add_update(aBound,T.switch(t/aBound_div>1,1.,0.01+t/aBound_div))
            self.add_update(aSH,T.switch(t/aSH_div>1,1.,0.01+t/aSH_div))

        # sharpening
        sharpening = self.add_weights('sharpening',np.asarray(self.params['sharpening']))
        self.add_update(sharpening,self.params['sharpening']*0.5*(1.+aSH))

        # save all hyperparameters to tOutputs for access later
        self.tOutputs.update(self.tWeights)
    

    def _buildModel(self):
        self.updates_ack = True
        self.tWeights = NestD(self.tWeights)
        self.tOutputs = NestD()
        """
        Build training, evaluation, inference, and sampling graphs for SemiVAE
        """

        #Inputs to graph
        XU = T.matrix('XU',   dtype=config.floatX)
        XL = T.matrix('XL',   dtype=config.floatX)
        YL = T.ivector('YL')
        epsU = T.matrix('epsU', dtype=config.floatX)
        epsL = T.matrix('epsL', dtype=config.floatX)
        self.fake_data(XU,XL,YL,epsU,epsL)


        # We will have a separate sub-namespace in tWeights for storing batchnorm statistics.
        # This is for two reasons:
        #  1) everything under self.tWeights['weights'] will be optimized against the training
        #     objective function, and so we want to keep the batchnorm stats out of there
        #  2) we will want to have separate batchnorm running statistics for each of the
        #     subgraphs p(x), p(x,y), and q(y|x); this is do to modeling convenience.
        with self.namespace('batchnorm_statistics','tWeights'):
            self.tBatchnormStats = self.tWeights

        """
        Build training graph
        """
        # set the namespace for tOutputs to 'train'
        # note that we would not want to set tWeights to 'train',
        # because the training and evaluating graphs share weights 
        with self.namespace('train','tOutputs'):
            # set tWeights and tOutputs to hyperparameters 
            # (these are adjusted during training, thus they go in the 'train' namespace)
            with self.namespaces('hyperparameters',['tWeights','tOutputs']):
                self.build_hyperparameters()
                self.tHyperparams = self.tOutputs

            # set the namespace for tWeights to 'weights'
            # (this is the same namespace for training, evaluation, and sampling)
            with self.namespace('weights','tWeights'):
                train_outputs = self.build_semi_vae(XU,XL,YL,epsU,epsL)
            
        """
        Build evaluation graph
        """
        # set the namespace for tOutputs to 'evaluate'
        # note that we would not want to set tWeights to 'evaluate',
        # because the training and evaluating graphs share weights 
        with self.namespace('evaluate','tOutputs'):
            # set the namespace for tWeights to 'weights'
            # (this is the same namespace for training, evaluation, and sampling)
            with self.namespace('weights','tWeights'):
                # set the entire class in evaluate mode 
                # i.e. this sets self._evaluating=True (instead of being False)
                with self.evaluate():
                    self.build_semi_vae(XU,XL,YL,epsU,epsL)
        
        """
        Build sampling graph
        """
        with self.namespace('samples','tOutputs'):
            with self.namespace('weights','tWeights'):
                with self.evaluate():
                    Z = T.matrix('Z',dtype=config.floatX)
                    alpha = T.matrix('alpha',dtype=config.floatX)
                    with self.namespaces('p(x)',['tOutputs','tBatchnormStats']):
                        self.tOutputs['probs'] = self.build_generative(alpha,Z)
                    with self.namespaces('p(x,y)',['tOutputs','tBatchnormStats']):
                        self.tOutputs['probs'] = self.build_generative(alpha,Z)
                

        #Training objective
        trainobjective = self.tOutputs['train']['objective'] 

        #Optimize all weights in self.tWeights['weights']
        #Note:
        # * tWeights is a NestD object, a nested dictionary with some helpful functions
        # * The .leaves() function returns all the leaves of a NestD object
        with self.namespace('weights','tWeights'):
            optimization_outputs = self.setup_optimizer(trainobjective,self.tWeights,
                                                        lr=self.tHyperparams['lr'],
                                                        optim_method=self.params['optimizer'], 
                                                        reg_type =self.params['reg_type'], 
                                                        reg_value= self.params['reg_value'],
                                                        grad_norm=self.params['grad_norm'],
                                                        divide_grad=self.params['divide_grad']
                                                        )
            optimizer_up, norm_list, trainobjective = optimization_outputs 

        print '\ntWeights:'
        self.print_namespace('tWeights')
        print '\ntOutputs:'
        self.print_namespace('tOutputs')
        print ''
        
        #save some optimization outputs
        with self.namespace('train','tOutputs'):
            self.tOutputs.update({'pnorm':norm_list[0],
                                  'gnorm':norm_list[1],
                                  'objective':trainobjective})

        #self.updates is container for all updates (e.g. see self.add_updates in AbstraceSemiVAE)
        self.updates += optimizer_up
        
        #Build theano functions
        fn_inputs = [XU,XL,YL,epsU,epsL]

        with self.namespace('train','tOutputs'):
            # .flatten(join='/') converts hierarchical NestD to NestD of depth 1
            # where keypaths are converted to string and joined by '/'
            # e.g. {'a':{'b':1}} becomes {'a/b':1}
            self.train = theano.function(fn_inputs,self.tOutputs.flatten(join='/')
                                        ,name='Train'
                                        ,updates=self.updates)

        with self.namespace('evaluate','tOutputs'):
            self.evaluate = theano.function(fn_inputs,self.tOutputs.flatten(join='/')
                                           ,name='Evaluate')

        # This should be called only once per epoch
        self.decay_lr = theano.function([],self.tHyperparams['lr'].sum()
                                       ,name='Update LR'
                                       ,updates=self.lr_update)

        with self.namespace('samples','tOutputs'):
            with self.namespace('p(x)','tOutputs'):
                self.sample_px = theano.function([Z,alpha],self.tOutputs['probs']
                                                ,name='sample p(x)')
            with self.namespace('p(x,y)','tOutputs'):
                self.sample_pxy = theano.function([Z,alpha],self.tOutputs['probs']
                                                  ,name='sample p(x)')

    def post_train_hook(self):
        self.decay_lr()

    def sample_model(self,nsamples=100):
        """
                                Sample from Generative Model
        """
        K = self.params['nclasses']
        z = np.random.randn(K*nsamples,self.params['dim_stochastic']).astype(config.floatX)
        alpha = np.repeat((np.arange(K).reshape(1,-1) == np.arange(K).reshape(-1,1)).astype('int'),axis=0,repeats=nsamples).astype(config.floatX)
        return {'U':self.sample_px(z,alpha),'L':self.sample_pxy(z,alpha)}

    def fake_data(self,XU,XL,Y,epsU,epsL):
        """
                                Compile all the fake data 
        """
        XU.tag.test_value = np.random.randint(0,2,(2, self.params['dim_observations'])).astype(config.floatX)
        XL.tag.test_value = np.random.randint(0,2,(20, self.params['dim_observations'])).astype(config.floatX)
        Y.tag.test_value = np.mod(np.arange(20),10).astype('int32')
        epsU.tag.test_value = np.random.randn(2, self.params['dim_stochastic']).astype(config.floatX)
        epsL.tag.test_value = np.random.randn(20, self.params['dim_stochastic']).astype(config.floatX)



