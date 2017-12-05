"""
This implements a version of the M2 model in:

Kingma, Diederik P., et al. "Semi-supervised learning with deep generative models." Advances in Neural Information Processing Systems. 2014.
"""

from ExactM2 import * 

class ApproxM2SemiVAE(ExactM2SemiVAE):

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

        # batchsize
        bs = eps.shape[0]
        if Y is not None: 
            """
            -logp(x,y)
            """
            # -log(p(y))
            nllY = bs*theano.shared(np.log(self.params['nclasses']))

            # gaussian parameters (Z)
            mu, logcov2 = self.build_inference_Z(Y,hx)

            # gaussian variates
            Z, KL_Z = self.variational_gaussian(mu,logcov2,eps)

            if not self._evaluating:
                # adding noise during training usually helps performance
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
            # categorical variates and KL(Y)
            probs = T.nnet.softmax(logbeta)
            logprobs = T.log(probs)
            KL_Y = T.sum(probs*logprobs,axis=1)+np.log(self.params['nclasses'])
            y = theano.gradient.disconnected_grad(self.srng.multinomial(n=1,pvals=probs,dtype=config.floatX))

            # gaussian parameters (Z)
            mu, logcov2 = self.build_inference_Z(y,hx)

            # gaussian variates
            Z, KL_Z = self.variational_gaussian(mu,logcov2,eps)

            if not self._evaluating:
                # adding noise during training usually helps performance
                Z = Z + self.srng.normal(Z.shape,0,0.05,dtype=config.floatX)

            # generative model
            paramsX = self.build_generative(y, Z)
            if self.params['data_type']=='real':
                nllX = self.nll_gaussian(X,**paramsX).sum(axis=1)
            else:
                nllX = self.nll_bernoulli(X,**paramsX).sum(axis=1)

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
            if Y is not None:
                objfunc = anneal['KL_Z']*KL_Z.sum() + NLL
            else:
                objfunc = anneal['KL_alpha']*KL_Y.sum() + anneal['KL_Z']*KL_Z.sum() + NLL

            # gradient hack to do black box variational inference:
            if Y is None: 
                # make sure sizes are correct to prevent unintentional broadcasting
                KL_Z = KL_Z.reshape([-1])
                nllX = nllX.reshape([-1])

                if self.params['negKL']:
                    # the negative KL trick is something we found by accident that
                    # works well for when alpha is assumed to be loggamma or dirichlet
                    f = theano.gradient.disconnected_grad(-2.*KL_Z+nllX)
                else:
                    f = theano.gradient.disconnected_grad(anneal['KL_Z']*KL_Z+nllX)

                # apply gradient hack to objective function
                logpdf = T.sum(y*logprobs,axis=1).reshape([-1])
                BBVIgradientHack = f*logpdf
                objfunc += BBVIgradientHack.sum()

        self.tOutputs.update({
                                'Z':Z,
                                'mu':mu,
                                'logcov2':logcov2,
                                'logbeta':logbeta,
                                'bound':bound,
                                'objfunc':objfunc,
                                'nllX':nllX,
                                'KL_Z':KL_Z,
                                'KL':KL,
                                'NLL':NLL,
                                'eps':eps,
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

