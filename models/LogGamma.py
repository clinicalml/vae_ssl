from AbstractSemiVAE import * 
from theano.compile.ops import as_op
from special import Psi, Polygamma
from randomvariates import randomLogGamma
import random
        
@as_op(itypes=[T.fmatrix,T.fscalar],otypes=[T.fmatrix])
def rng_loggamma_(beta,seed):
    vfunc = np.vectorize(randomLogGamma)
    random.seed(float(seed))
    return vfunc(beta).astype(config.floatX)

class LogGammaSemiVAE(AbstractSemiVAE):

    def rng_loggamma(self, beta):
        seed=self.srng.uniform(size=(1,),low=-1.0e10,high=1.0e10)[0]
        return rng_loggamma_(beta,seed)

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


    def _build_generative(self, alpha, Z):
        """
        Build subgraph to estimate conditional params
        """
        with self.namespaces('p(x|z,alpha)'):

            # first transform Z into another embedding layer
            with self.namespaces('h(z)'):        
                Z = self._buildHiddenLayers(Z,diminput=self.params['dim_stochastic']
                                             ,dimoutput=self.params['p_dim_hidden']
                                             ,nlayers=self.params['z_generative_layers']
                                             ,normalization=self.params['p_normlayers'])

            # combine alpha and Z
            h = T.concatenate([alpha,Z],axis=1)

            # hidden layers for p(x|z,alpha)
            with self.namespaces('h(h(z),alpha)'):        
                h = self._buildHiddenLayers(h,diminput=self.params['dim_stochastic']+self.params['nclasses']
                                             ,dimoutput=self.params['p_dim_hidden']
                                             ,nlayers=self.params['p_layers']
                                             ,normalization=self.params['p_normlayers'])

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
        accuracy = T.eq(T.argmax(probs,axis=1),Y)
        return probs, loss, accuracy

    def _build_hx_logbeta(self, X): 
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

    def _buildVAE(self, X, eps, Y=None):
        """
        Build VAE subgraph to do inference and emissions 
        (if Y==None, build upper bound of -logp(x), else build upper bound of -logp(x,y)

        returns a bunch of VAE outputs
        """
        betaprior = self.tHyperparams['betaprior']
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
        #    * we have separate batchnorm running stats for each of p(x), p(x,y), q(y|x)
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

            mu, logcov2 = self._build_inference(alpha,hx)

            # gaussian variates
            Z, KL_Z = self._variationalGaussian(mu,logcov2,eps)

            if not self._evaluating:
                # adding noise during training usually helps performance
                Z = Z + self.srng.normal(Z.shape,0,0.05,dtype=config.floatX)

            # generative model
            paramsX = self._build_generative(alpha, Z)
            if self.params['data_type']=='real':
                nllX = self._nll_gaussian(X,**paramsX).sum(axis=1)
            else:
                nllX = self._nll_bernoulli(X,**paramsX).sum(axis=1)

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


    def _setupHyperparameters(self):
        super(LogGammaSemiVAE,self)._setupHyperparameters()

        t = self.tWeights['update_ctr']
        with self.namespaces('annealing'):
            # annealing parameters
            aBP = self._addWeights('betaprior',np.asarray(0.))
            aBP_div = float(self.params['annealBP']) #50000.

            # updates
            self._addUpdate(aBP,T.switch(t/aBP_div>1,1.,0.01+t/aBP_div))

        # betaprior is annealed from self.params['betaprior'] to self.params['finalbeta']
        finalbeta  = float(self.params['finalbeta'])
        betaprior = self.params['betaprior']*(1-aBP)+self.params['finalbeta']*aBP

        # save all hyperparameters to tOutputs for access later
        self.tOutputs.update(self.tWeights)
        self.tOutputs['betaprior'] = betaprior

    def progressBarReportMap(self):
        # see self.progressBarUpdate for use
        # use list to preserve order
        report_map = super(LogGammaSemiVAE,self).progressBarReportMap()
        return report_map +[
            ('hyperparameters/betaprior',lambda x:np.mean(x[-1]),'%05.f (last)'),
        ]
    
