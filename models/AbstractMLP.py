from AbstractModel import * 

class AbstractMLP(AbstractModel):

    def __init__(self,params,*args,**kwargs):
        params['data_type'] = 'binary'
        super(AbstractMLP,self).__init__(params,*args,**kwargs)

    def sample_dataset(self, dataset):
        p = np.random.uniform(low=0,high=1,size=dataset.shape)
        return (dataset >= p).astype(config.floatX)

    def preprocess_minibatch(self,minibatch):
        nL = len(minibatch['L']['X'])
        return {
                'X':self.sample_dataset(minibatch['L']['X']),
                'Y':minibatch['L']['Y'].astype('int32'),
               }

    def progress_bar_report_map(self):
        # see self.progress_bar_update for use
        # use list to preserve order
        return [
            ('accuracy',lambda x:x.astype(float).mean(),'%0.2f (epoch mean)'),
            ('loss',np.mean,'%0.2f (epoch mean)'),
            ('classifier',np.mean,'%0.2f (epoch mean)'),
            ('hyperparameters/lr',lambda x:np.mean(x[-1]),'%0.2e (last)'),
        ]

    def build_classifier(self, X, Y):
        """
        calculate -log(q(Y|X))

        return probs, loss, accuracy
        """
        pass

    def build_mlp(self,X,Y):
        """
        build a simple MLP similar to that in AbstractSemiVAE
        """

        # from now on, under this with statement, we will almost always be working
        # with the same set of namespaces, unless otherwise stated, so lets set them as defaults.
        # (this is actually already the default, but let's put it here anyway)
        #
        # tWeights: container for all theano shared variables
        # tOutputs: container for variables that will be outputs of our theano functions
        # tBatchnormStats: container for batchnorm running statistics
        with self.set_attr('default_namespaces',['tWeights','tOutputs','tBatchnormStats']):

            # set tOutputs and tBatchnormStats namespaces to 'q(y|x)'
            with self.namespaces('q(y|x)',['tOutputs','tBatchnormStats']):
                _, crossentropyloss, accuracy = self.build_classifier(X,Y)

        # calculate bound, loss, and theano-specific objective function (w/ gradient hacks)
        # keep classifier weight here for comparison with SemiVAE's 
        classifier = self.params['classifier_weight']*crossentropyloss.sum()
        objective = classifier
        loss = classifier

        # for reporting purposes, normalize some of the outputs
        self.tOutputs.update({
                                'classifier':classifier/X.shape[0],
                                'loss':loss/X.shape[0],
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
        X = T.matrix('X',   dtype=config.floatX)
        Y = T.ivector('Y')
        self.fake_data(X,Y)


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
                self.build_mlp(X,Y)
            
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
                    self.build_mlp(X,Y)
        
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
        fn_inputs = [X,Y]

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

    def post_train_hook(self,**kwargs):
        self.decay_lr()

    def post_valid_hook(self,**kwargs):
        pass

    def post_test_hook(self,**kwargs):
        pass

    def post_save_hook(self,**kwargs):
        pass

    def fake_data(self,X,Y):
        """
                                Compile all the fake data 
        """
        X.tag.test_value = np.random.randint(0,2,(20, self.params['dim_observations'])).astype(config.floatX)
        Y.tag.test_value = np.mod(np.arange(20),10).astype('int32')



