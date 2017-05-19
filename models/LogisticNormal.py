from AbstractSemiVAE import * 

class LogisticNormalSemiVAE(AbstractSemiVAE):

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

    def _build_inference_alpha(self, X): 
        """
        return h(x), mu(h(x)), logcov2(h(x))
        """
        if not self._evaluating:
            X = self._dropout(X,self.params['input_dropout'])
            self._p(('Inference with dropout :%.4f')%(self.params['input_dropout']))


        with self.namespaces('h(x)'):
            hx = self._buildHiddenLayers(X,diminput=self.params['dim_observations']
                                          ,dimoutput=self.params['q_dim_hidden']
                                          ,nlayers=self.params['q_layers'])

        with self.namespaces('mu(h(x)), logcov2(h(x))'):
            h_alpha = self._buildHiddenLayers(hx,diminput=self.params['q_dim_hidden']
                                                ,dimoutput=self.params['q_dim_hidden']
                                                ,nlayers=self.params['alpha_inference_layers'])

            if not self._evaluating:
                h_alpha = self._dropout(h_alpha,self.params['dropout_logbeta']) 

            with self.namespaces('mu'):
                mu = self._linear(h_alpha,diminput=self.params['q_dim_hidden']
                                         ,dimoutput=self.params['nclasses'])

            with self.namespaces('logcov2'):
                logcov2 = self._linear(h_alpha,diminput=self.params['q_dim_hidden']
                                              ,dimoutput=self.params['nclasses'])

        self.tOutputs['mu'] = mu
        self.tOutputs['logcov2'] = logcov2
        return hx, mu, logcov2

    def _build_inference_Z(self,alpha,hx):
        """
        return q(z|alpha,h(x))
        """

        if not self._evaluating:
            hx = self._dropout(hx,self.params['dropout_hx'])

        with self.namespaces('hz(hx)'):
            hz = self._buildHiddenLayers(hx,diminput=self.params['q_dim_hidden']
                                           ,dimoutput=self.params['q_dim_hidden']
                                           ,nlayers=self.params['hz_inference_layers'])

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

            with self.namespaces('mu'):
                mu = self._linear(q_Z_h,diminput=self.params['q_dim_hidden']
                                       ,dimoutput=self.params['dim_stochastic'])
            with self.namespaces('logcov2'):
                logcov2 = self._linear(q_Z_h,diminput=self.params['q_dim_hidden']
                                              ,dimoutput=self.params['dim_stochastic'])

        return mu, logcov2

    def _buildVAE(self, X, eps, Y=None):
        """
        Build VAE subgraph to do inference and emissions 
        (if Y==None, build upper bound of -logp(x), else build upper bound of -logp(x,y)

        returns a bunch of VAE outputs
        """
        if Y is None:
            self._p(('Building graph for lower bound of logp(x)'))
        else:
            self._p(('Building graph for lower bound of logp(x,y)'))

        # build h(x) and mu_alpha, logcov2_alpha
        hx, mu_alpha, logcov2_alpha = self._build_inference_alpha(X)

        if Y is not None: 
            """
            -logp(x,y)
            """
            if self.params['learn_posterior']:
                with self.namespaces('q(alpha|y)'):            
                    mu_alpha += T.nnet.softplus(self._addWeights('mu_alpha|y',np.asarray(1.)))*Y
                    logcov2_alpha += self._linear(self._addWeights('logcov2_alpha|y',np.asarray(1.)))*Y
            else:
                mu_alpha += T.nnet.softplus(self.params['posterior_val'])*Y

            # gaussian variates
            eps_alpha = self.srng.normal(mu_alpha.shape,1,dtype=config.floatX) 
            logit_alpha, KL_alpha = self._variationalGaussian(mu_alpha,logcov2_alpha,eps_alpha)

            # derive logistic-normal
            alpha = T.nnet.softmax(logit_alpha*self.tHyperparams['sharpening'])

            # -log(p(y|alpha))
            nllY = T.nnet.categorical_crossentropy(alpha,Y)

        else:
            """
            -logp(x)
            """
            # gaussian variates
            eps_alpha = self.srng.normal(mu_alpha.shape,1,dtype=config.floatX) 
            logit_alpha, KL_alpha = self._variationalGaussian(mu_alpha,logcov2_alpha,eps_alpha)

            # derive logistic-normal
            alpha = T.nnet.softmax(logit_alpha*self.tHyperparams['sharpening'])

        # parameters of Z 
        mu, logcov2 = self._build_inference_Z(alpha,hx)

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
        KL = KL_alpha.sum() + KL_Z.sum()
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
            objfunc = anneal['KL_alpha']*KL_alpha.sum() + anneal['KL_Z']*KL_Z.sum() + NLL

        self.tOutputs.update({
                                'alpha':alpha,
                                'logit_alpha':logit_alpha,
                                'Z':Z,
                                'mu_alpha':mu_alpha,
                                'logcov2_alpha':logcov2_alpha,
                                'mu':mu,
                                'logcov2':logcov2,
                                'bound':bound,
                                'objfunc':objfunc,
                                'nllX':nllX,
                                'KL_alpha':KL_alpha,
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

    def _build_classifier(self, XL, Y):
        _, mu_alpha, logcov2_alpha = self._build_inference_alpha(XL)
        if self._evaluating:
            # this is a hack, but it works because we generally only
            # care about accuracy when evaluating semi-supervised MNIST
            alpha = T.nnet.softmax(mu_alpha)
        else:
            # gaussian variates
            eps_alpha = self.srng.normal(mu_alpha.shape,1,dtype=config.floatX) 
            logit_alpha, KL_alpha = self._variationalGaussian(mu_alpha,logcov2_alpha,eps_alpha)

            # derive logistic-normal
            alpha = T.nnet.softmax(logit_alpha*self.tHyperparams['sharpening'])

        probs = alpha
        #T.nnet.categorical_crossentropy returns a vector of length batch_size
        loss= T.nnet.categorical_crossentropy(probs,Y) 
        accuracy = T.eq(T.argmax(probs,axis=1),Y)
        return probs, loss, accuracy

