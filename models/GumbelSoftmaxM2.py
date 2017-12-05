"""
This implements a version of the Gumbel Softmax model in:

Jang, Eric, Shixiang Gu, and Ben Poole. "Categorical reparameterization with gumbel-softmax." arXiv preprint arXiv:1611.01144 (2016).

This model is based off of the M2 model in:

Kingma, Diederik P., et al. "Semi-supervised learning with deep generative models." Advances in Neural Information Processing Systems. 2014.
"""

from ExactM2 import * 

class GumbelSoftmaxM2SemiVAE(ExactM2SemiVAE):

    def sample_y(self,logbeta):
        u = self.srng.uniform(logbeta.shape,low=1e-10,high=1.-1e-10,dtype=config.floatX)
        u = theano.gradient.disconnected_grad(u)
        g = -T.log(-T.log(u))
        if self.params['model']=='GumbelSoftmaxM2':
            y = T.nnet.softmax((g+logbeta)*self.tHyperparams['sharpening'])
        elif self.params['model']=='STGumbelSoftmaxM2':
            y = T.nnet.softmax((g+logbeta)*self.tHyperparams['sharpening'])
            y_discrete = T.argmax(g+logbeta,axis=1)
            y_discrete = T.extra_ops.to_one_hot(y_discrete,self.params['nclasses'],dtype=config.floatX)
            y = theano.gradient.disconnected_grad(y_discrete-y)+y
        else:
            assert False, 'unhandled GumbelSoftmax type %s' % self.params['model']
        return y


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
            y = self.sample_y(logbeta)

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

