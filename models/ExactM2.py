from AbstractSingleStochasticLayerSemiVAE import * 
        
class ExactM2SemiVAE(AbstractSingleStochasticLayerSemiVAE):

    def build_inference_Y(self, X): 
        """
        return h(x), logbeta(h(x))
        """
        if not self._evaluating:
            X = self.dropout(X,self.params['input_dropout'])
            self._p(('Inference with dropout :%.4f')%(self.params['input_dropout']))


        with self.namespaces('h(x)'):
            hx = self.build_hidden_layers(X,diminput=self.params['dim_observations']
                                          ,dimoutput=self.params['q_dim_hidden']
                                          ,nlayers=self.params['q_layers'])

        with self.namespaces('h_logbeta'):
            h_logbeta = self.build_hidden_layers(hx,diminput=self.params['q_dim_hidden']
                                                  ,dimoutput=self.params['q_dim_hidden']
                                                  ,nlayers=self.params['alpha_inference_layers'])

        if not self._evaluating:
            h_logbeta = self.dropout(h_logbeta,self.params['dropout_logbeta']) 

        with self.namespaces('logbeta'):
            logbeta = self.linear(h_logbeta,diminput=self.params['q_dim_hidden']
                                            ,dimoutput=self.params['nclasses'])

        #clip to avoid nans
        logbeta = T.clip(logbeta,-5,5)

        self.tOutputs['logbeta'] = logbeta
        return hx, logbeta

    def build_vae(self, X, eps, Y=None):
        """
        Build VAE subgraph to do inference and emissions 
        (if Y==None, build upper bound of -logp(x), else build upper bound of -logp(x,y)

        returns a bunch of VAE outputs
        """
        if Y is None:
            self._p(('Building graph for lower bound of logp(x)'))
        else:
            self._p(('Building graph for lower bound of logp(x,y)'))

        # build h(x) and logbeta
        hx, logbeta = self.build_inference_Y(X)

        #we don't actually use eps here, but do the following to include it in graph computation
        bs = eps.shape[0]

        if Y is not None: 
            """
            -logp(x,y)
            """
            nllY = bs*theano.shared(np.log(self.params['nclasses']))

            # gaussian parameters (Z)
            mu, logcov2 = self.build_inference_Z(Y,hx)

            # gaussian variates
            eps = self.srng.normal(mu.shape,0,1,dtype=config.floatX) 
            Z, KL_Z = self.variational_gaussian(mu,logcov2,eps)

            # adding noise during training usually helps performance
            if not self._evaluating:
                Z = Z + self.srng.normal(Z.shape,0,0.05,dtype=config.floatX)

            # generative model
            paramsX = self.build_generative(Y, Z)
            if self.params['data_type']=='real':
                nllX = self.nll_gaussian(X,**paramsX).sum(axis=1)
            else:
                nllX = self.nll_bernoulli(X,**paramsX).sum(axis=1)

            KL = KL_Z.sum()
            NLL = nllX.sum() + nllY.sum()
        else:
            """
            -logp(x)
            """
            # KL of categorical
            probs = T.nnet.softmax(logbeta)
            logprobs = T.log(probs)
            KL_Y = T.sum(probs*logprobs,axis=1) + np.log(self.params['nclasses'])

            # Enumerate all classes 
            y = [T.extra_ops.to_one_hot(T.extra_ops.repeat(theano.shared(i),repeats=hx.shape[0]),nb_class=self.params['nclasses']) for i in range(self.params['nclasses'])]
            y = T.concatenate(y,axis=0)

            # repeat for each class (do this to preserve batchnorm stats
            # over the class enumeration)
            hx_repeat = T.tile(hx.T,self.params['nclasses'],ndim=2).T
            X_repeat = T.tile(X.T,self.params['nclasses'],ndim=2).T

            # gaussian parameters (Z)
            mu, logcov2 = self.build_inference_Z(y,hx_repeat)

            # gaussian variates
            eps = self.srng.normal(mu.shape,0,1,dtype=config.floatX) 
            Z, KL_Z = self.variational_gaussian(mu,logcov2,eps)

            # adding noise during training usually helps performance
            if not self._evaluating:
                Z = Z + self.srng.normal(Z.shape,0,0.05,dtype=config.floatX)

            # generative model
            paramsX = self.build_generative(y, Z)
            if self.params['data_type']=='real':
                nllX = self.nll_gaussian(X_repeat,**paramsX).sum(axis=1)
            else:
                nllX = self.nll_bernoulli(X_repeat,**paramsX).sum(axis=1)

            # evaluate kl and nll over all states of Y
            kl = 0
            nll = 0 
            for c in range(self.params['nclasses']):
                start_idx = c*bs
                end_idx = (c+1)*bs
                kl += 2*probs[:,c]*KL_Z[start_idx:end_idx].ravel()
                nll += probs[:,c]*nllX[start_idx:end_idx].ravel()
                #kl += probs[:,c]*KL_Z[start_idx:end_idx].ravel()
            KL_Z = kl
            nllX = nll
            KL = KL_Y.sum() + KL_Z.sum()
            NLL = nllX.sum() 

        bound = KL + NLL 

        # objective function
        if self._evaluating:
            objfunc = bound 
        else: 
            # annealing (training only)
            anneal = self.tHyperparams['annealing']

            # annealed objective function
            if Y is None:
                objfunc = anneal['KL_alpha']*KL_Y.sum() + anneal['KL_Z']*KL_Z.sum() + NLL
            else:
                objfunc = anneal['KL_Z']*KL_Z.sum() + NLL

        self.tOutputs.update({
                                'Z':Z,
                                'mu':mu,
                                'logcov2':logcov2,
                                'bound':bound,
                                'objfunc':objfunc,
                                'nllX':nllX,
                                'KL_Z':KL_Z,
                                'KL':KL,
                                'NLL':NLL,
                             })
        if Y is not None:
            self.tOutputs.update({
                                'nllY':nllY
                                })
        else:
            self.tOutputs.update({
                                'KL_Y':KL_Y
                                })

        return self.tOutputs

    def build_classifier(self, XL, Y):
        _, logbeta = self.build_inference_Y(XL)
        probs = T.nnet.softmax(logbeta)
        #T.nnet.categorical_crossentropy returns a vector of length batch_size
        loss= T.nnet.categorical_crossentropy(probs,Y) 
        accuracy = T.eq(T.argmax(probs,axis=1),Y)
        return probs, loss, accuracy

