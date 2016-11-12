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
from theanomodels.models import BaseModel
from vae_ssl_LogGamma import LogGammaSemiVAE
import scipy
import ipdb


def logsumexp(logs):
    maxlog = T.max(logs,axis=1,keepdims=True)
    return maxlog + T.log(T.sum(T.exp(logs-maxlog),axis=1,keepdims=True))

class LogisticNormalSemiVAE(LogGammaSemiVAE):

    def __init__(self,params,paramFile=None,reloadFile=None):
        super(LogisticNormalSemiVAE,self).__init__(params,paramFile=paramFile,reloadFile=reloadFile)

    def _createParams(self):
        """
                    _createParams: create parameters necessary for the model
        """
        npWeights = OrderedDict()
        if 'q_dim_hidden' not in self.params or 'p_dim_hidden' not in self.params:
            self.params['q_dim_hidden']= dim_hidden
            self.params['p_dim_hidden']= dim_hidden
        DIM_HIDDEN = self.params['q_dim_hidden']
        #Weights in recognition network model
        for q_l in range(self.params['q_layers']):
            dim_input     = DIM_HIDDEN
            dim_output    = DIM_HIDDEN
            if q_l==0:
                dim_input     = self.params['dim_observations']
            if self.params['nonlinearity']=='maxout':
                dim_output= DIM_HIDDEN*self.params['maxout_stride']
            # h(x)
            npWeights['q_h(x)_'+str(q_l)+'_W'] = self._getWeight((dim_input, dim_output))
            npWeights['q_h(x)_'+str(q_l)+'_b'] = self._getWeight((dim_output, ))
        for a_l in range(self.params['alpha_inference_layers']):
            dim_input     = DIM_HIDDEN
            dim_output    = DIM_HIDDEN
            if self.params['nonlinearity']=='maxout':
                dim_output= DIM_HIDDEN*self.params['maxout_stride']
            # log(beta)
            npWeights['q_hz2_'+str(a_l)+'_W'] = self._getWeight((dim_input, dim_output))
            npWeights['q_hz2_'+str(a_l)+'_b'] = self._getWeight((dim_output, ))
        for hz_l in range(self.params['hz_inference_layers']):
            dim_input = DIM_HIDDEN
            dim_output = DIM_HIDDEN
            if self.params['nonlinearity']=='maxout':
                dim_output = DIM_HIDDEN*self.params['maxout_stride']
            npWeights['q_hz_%s_W' % hz_l] = self._getWeight((dim_input, dim_output))
            npWeights['q_hz_%s_b' % hz_l] = self._getWeight((dim_output, ))
        for z_l in range(self.params['z_inference_layers']):
            dim_input     = DIM_HIDDEN
            if z_l == 0 and not self.params['bilinear']:
                dim_input = 2*DIM_HIDDEN
            dim_output    = DIM_HIDDEN
            if self.params['nonlinearity']=='maxout':
                dim_output= DIM_HIDDEN*self.params['maxout_stride']
            # Z
            npWeights['q_Z_'+str(z_l)+'_W'] = self._getWeight((dim_input, dim_output))
            npWeights['q_Z_'+str(z_l)+'_b'] = self._getWeight((dim_output, ))

        if self.params['inference_model']=='single':
            npWeights['q_hz2_mu_W'] = self._getWeight((DIM_HIDDEN, self.params['nclasses']))
            npWeights['q_hz2_mu_b'] = self._getWeight((self.params['nclasses'],))
            npWeights['q_hz2_logcov2_W'] = self._getWeight((DIM_HIDDEN, self.params['nclasses']))
            npWeights['q_hz2_logcov2_b'] = self._getWeight((self.params['nclasses'],))
            npWeights['q_Z2_mu_W|y'] = self._getWeight((self.params['nclasses'],self.params['nclasses']))
            npWeights['q_Z2_logcov2_W|y'] = self._getWeight((self.params['nclasses'],self.params['nclasses']))
            dim_output = DIM_HIDDEN
            if self.params['nonlinearity']=='maxout':
                dim_output= DIM_HIDDEN*self.params['maxout_stride']
            if self.params['bilinear']:
                npWeights['q_alpha_h(x)_W'] = self._getWeight((DIM_HIDDEN,DIM_HIDDEN,self.params['nclasses']))
            else:
                npWeights['q_alpha_h(x)_W'] = self._getWeight((self.params['nclasses'],DIM_HIDDEN))
            npWeights['q_alpha_h(x)_b'] = self._getWeight((DIM_HIDDEN,))
            #npWeights['q_Z_hx_alpha_W'] = self._getWeight((self.params['nclasses']+DIM_HIDDEN,dim_output))
            #npWeights['q_Z_hx_alpha_b'] = self._getWeight((dim_output,))
            npWeights['q_Z_mu_W']     = self._getWeight((DIM_HIDDEN, self.params['dim_stochastic']))
            npWeights['q_Z_mu_b']     = self._getWeight((self.params['dim_stochastic'],))
            npWeights['q_Z_logcov_W'] = self._getWeight((DIM_HIDDEN, self.params['dim_stochastic']))
            npWeights['q_Z_logcov_b'] = self._getWeight((self.params['dim_stochastic'],))

        else:
            assert False,'Invalid variational model'


        #Generative Model
        DIM_HIDDEN = self.params['p_dim_hidden']
        dim_input = self.params['dim_stochastic']
        for pz_l in range(self.params['z_generative_layers']):
            dim_output = DIM_HIDDEN
            if self.params['nonlinearity']=='maxout':
                dim_output = DIM_HIDDEN*self.params['maxout_stride']
            npWeights['p_z_%s_W' % pz_l] = self._getWeight((dim_input, dim_output))
            npWeights['p_z_%s_b' % pz_l] = self._getWeight((dim_output, ))
            dim_input = DIM_HIDDEN
        if self.params['bilinear']:
            npWeights['p_alpha_Z_input_W'] = self._getWeight((DIM_HIDDEN,dim_input,self.params['nclasses']))
            dim_input = DIM_HIDDEN
        else:
            dim_input += self.params['nclasses']
        for p_l in range(self.params['p_layers']):
            dim_output    = DIM_HIDDEN
            if self.params['nonlinearity']=='maxout':
                dim_output= DIM_HIDDEN*self.params['maxout_stride']
            npWeights['p_'+str(p_l)+'_W'] = self._getWeight((dim_input, dim_output))
            npWeights['p_'+str(p_l)+'_b'] = self._getWeight((dim_output, ))
            dim_input     = DIM_HIDDEN
        if self.params['data_type']=='real':
            npWeights['p_mu_W']     = self._getWeight((DIM_HIDDEN, self.params['dim_observations']))
            npWeights['p_logcov_W'] = self._getWeight((DIM_HIDDEN, self.params['dim_observations']))
            npWeights['p_mu_b']     = self._getWeight((self.params['dim_observations'],))
            npWeights['p_logcov_b'] = self._getWeight((self.params['dim_observations'],))
        else:
            npWeights['p_mean_W']     = self._getWeight((DIM_HIDDEN, self.params['dim_observations']))
            npWeights['p_mean_b']     = self._getWeight((self.params['dim_observations'],))
        return npWeights
        
        
    def _buildRecognitionBase(self, X, dropout_prob=0.,evaluation=False,modifiedBatchNorm=False,graphprefix=None):
        if self.params['modifiedBatchNorm']:
            self._BNprefix=None
        elif self.params['separateBNrunningstats']:
            self._BNprefix=graphprefix
        else:
            self._BNprefix=None
        self._p(('Inference with dropout :%.4f')%(dropout_prob))
        inp = self._dropout(X,dropout_prob)
        hx = self._buildHiddenLayers(inp,self.params['q_layers'],'q_h(x)_{layer}',evaluation,modifiedBatchNorm=modifiedBatchNorm)
        hz2 = self._buildHiddenLayers(hx,self.params['alpha_inference_layers'],'q_hz2_{layer}',evaluation,modifiedBatchNorm=modifiedBatchNorm)
        if evaluation==False and self.params['dropout_logbeta'] > 0:
            hz2 = self._dropout(hz2,self.params['dropout_logbeta'])
        mu = self._LinearNL(W=self.tWeights['q_hz2_mu_W'],b=self.tWeights['q_hz2_mu_b'],inp=hz2,onlyLinear=True)
        logcov2 = self._LinearNL(W=self.tWeights['q_hz2_logcov2_W'],b=self.tWeights['q_hz2_logcov2_b'],inp=hz2,onlyLinear=True)
        return hx, mu, logcov2 

    def _build_qz(self,alpha,hx,evaluation,modifiedBatchNorm,graphprefix):
        if self.params['dropout_hx']>0 and evaluation==False:
            hx = self._dropout(hx,self.params['dropout_hx'])
        hz = self._buildHiddenLayers(hx,self.params['hz_inference_layers'],'q_hz_{layer}',evaluation,modifiedBatchNorm=modifiedBatchNorm)
        if self.params['separateBNrunningstats']:
            self._BNprefix=graphprefix
            modifiedBatchNorm=False
        else:
            self._BNprefix=None
        suffix = '_e' if evaluation else '_t'
        #merge h(x) and alpha
        if self.params['bilinear']:
            self._addVariable(graphprefix+'_hz'+suffix,hz,True)
            hz_W_alpha = self._bilinear(hz,self.tWeights['q_alpha_h(x)_W'],alpha)
            self._addVariable(graphprefix+'_hz_W_alpha'+suffix,hz_W_alpha,True)
            hz_alpha = T.nnet.softplus(hz_W_alpha)
        else:
            alpha_embed = self._LinearNL(W=self.tWeights['q_alpha_h(x)_W'],b=self.tWeights['q_alpha_h(x)_b'],inp=alpha,onlyLinear=True)
            self._addVariable(graphprefix+'_alpha_embed'+suffix,alpha_embed,True)
            hz_alpha = T.concatenate([alpha_embed,hz],axis=1)
        #infer mu and logcov
        q_Z_hidden = self._buildHiddenLayers(hz_alpha,self.params['z_inference_layers'],'q_Z_{layer}',evaluation,modifiedBatchNorm=modifiedBatchNorm)
        mu      = self._LinearNL(self.tWeights['q_Z_mu_W'],self.tWeights['q_Z_mu_b'],q_Z_hidden,onlyLinear=True)
        logcov  = self._LinearNL(self.tWeights['q_Z_logcov_W'],self.tWeights['q_Z_logcov_b'],q_Z_hidden,onlyLinear=True)
        return mu, logcov

    def _buildGraph(self, X, eps, betaprior=0.2, Y=None, dropout_prob=0.,add_noise=False,annealKL=None,evaluation=False,modifiedBatchNorm=False,graphprefix=None):
        """
                                Build VAE subgraph to do inference and emissions
                (if Y==None, build upper bound of -logp(x), else build upper bound of -logp(x,y)
        """
        if Y == None:
            self._p(('Building graph for lower bound of logp(x)'))
        else:
            self._p(('Building graph for lower bound of logp(x,y)'))
        hx, mu_Z2,logcov2_Z2= self._buildRecognitionBase(X,dropout_prob,evaluation,modifiedBatchNorm,graphprefix)
        suffix = '_e' if evaluation else '_t'
        self._addVariable(graphprefix+'_h(x)'+suffix,hx,ignore_warnings=True)

        if Y!=None:
            mu_Z2 += self._LinearNL(self.tWeights['q_Z2_mu_W|y'],0,Y,onlyLinear=True)
            logcov2_Z2 += self._LinearNL(self.tWeights['q_Z2_logcov2_W|y'],0,Y,onlyLinear=True)
        self._addVariable(graphprefix+'_Z2_mu'+suffix,mu_Z2,ignore_warnings=True)
        self._addVariable(graphprefix+'_Z2_logcov2'+suffix,logcov2_Z2,ignore_warnings=True)
        eps2 = self.srng.normal(mu_Z2.shape,1,dtype=config.floatX)
        Z2, KL_Z2 = self._variationalGaussian(mu_Z2,logcov2_Z2,eps2)
        if add_noise:
            Z2 = Z2 + self.srng.normal(Z2.shape,0,0.05,dtype=config.floatX)
        if self.params['no_softmax']:
            alpha = Z2
        else:
            if self.params['sharpening'] != 1:
                alpha = T.nnet.softmax(Z2*self.params['sharpening'])
            else:
                alpha = T.nnet.softmax(Z2)
        if Y!=None:
            nllY = T.nnet.categorical_crossentropy(T.nnet.softmax(Z2),Y)
        
        mu, logcov = self._build_qz(alpha,hx,evaluation,modifiedBatchNorm,graphprefix)
        self._addVariable(graphprefix+'_mu'+suffix,mu,ignore_warnings=True)
        self._addVariable(graphprefix+'_logcov2'+suffix,logcov,ignore_warnings=True)
        #Z = gaussian variates
        Z, KL_Z = self._variationalGaussian(mu,logcov,eps)
        if add_noise:
            Z = Z + self.srng.normal(Z.shape,0,0.05,dtype=config.floatX)
        #generative model
        _, nllX = self._buildEmission(alpha, Z, X, graphprefix, evaluation=evaluation,modifiedBatchNorm=modifiedBatchNorm)

        #negative of the lower bound
        if self.params['KL_loggamma_coef'] != 1:
            KL = self.params['KL_loggamma_coef']*KL_Z2.sum() + KL_Z.sum()
        else:
            KL = KL_Z2.sum() + KL_Z.sum()
        NLL = nllX.sum()
        if Y!=None:
            NLL += nllY.sum()
        bound = KL + NLL
        #objective function
        if evaluation:
            objfunc = bound
        else:
            if annealKL == None:
                objfunc = bound
            else:
                objfunc = annealKL*KL + NLL

        outputs = {'alpha':alpha,
                   'Z':Z,
                   'Z2':Z2,
                   'bound':bound,
                   'objfunc':objfunc,
                   'nllX':nllX,
                   'KL_Z2':KL_Z2,
                   'KL_Z':KL_Z,
                   'KL':KL,
                   'NLL':NLL}
        if Y!=None:
            outputs['nllY']=nllY

        return outputs

    def _buildClassifier(self, X, Y, add_noise, dropout_prob, evaluation, modifiedBatchNorm, graphprefix):
        hx, mu_Z2,logcov2_Z2= self._buildRecognitionBase(X,dropout_prob,evaluation,modifiedBatchNorm,graphprefix)
        if evaluation:
            alpha = T.nnet.softmax(mu_Z2)
        else:
            eps2 = self.srng.normal(mu_Z2.shape,1,dtype=config.floatX)
            Z2, KL_Z2 = self._variationalGaussian(mu_Z2,logcov2_Z2,eps2)
            if add_noise:
                Z2 = Z2 + self.srng.normal(Z2.shape,0,0.05,dtype=config.floatX)
            if self.params['sharpening'] != 1:
                alpha = T.nnet.softmax(Z2*self.params['sharpening'])
            else:
                alpha = T.nnet.softmax(Z2)
        #T.nnet.categorical_crossentropy returns a vector of length batch_size
        probs = alpha
        loss= T.nnet.categorical_crossentropy(probs,Y)
        ncorrect = T.eq(T.argmax(probs,axis=1),Y).sum()
        return probs, loss, ncorrect

    def _getModelOutputs(self,outputsU,outputsL,suffix=''):
        my_outputs = {
                         'KL_Z2_U':outputsU['KL_Z2'].sum(),
                         'nllX_U':outputsU['nllX'].sum(),
                         'NLL_U':outputsU['NLL'].sum(),
                         'KL_U':outputsU['KL'].sum(),
                         'KL_Z_U':outputsU['KL_Z'].sum(),
                         'nllY':outputsL['nllY'].sum(),
                         'KL_Z2_L':outputsL['KL_Z2'].sum(),
                         'nllX_L':outputsL['nllX'].sum(),
                         'NLL_L':outputsL['NLL'].sum(),
                         'KL_L':outputsL['KL'].sum(),
                         'KL_Z_L':outputsL['KL_Z'].sum(),
                         'mu_Z2_U':self._tVariables['U_Z2_mu'+suffix],
                         'mu_Z2_L':self._tVariables['L_Z2_mu'+suffix],
                         'logcov2_Z2_U':self._tVariables['U_Z2_logcov2'+suffix],
                         'logcov2_Z2_L':self._tVariables['L_Z2_logcov2'+suffix],
                         'mu_U':self._tVariables['U_mu'+suffix],
                         'mu_L':self._tVariables['L_mu'+suffix],
                         'logcov2_U':self._tVariables['U_logcov2'+suffix],
                         'logcov2_L':self._tVariables['L_logcov2'+suffix],
                         'q_Z2_mu_W|y':self.tWeights['q_Z2_mu_W|y'],
                         'q_Z2_logcov2_W|y':self.tWeights['q_Z2_logcov2_W|y'],
                         }
        if self.params['bilinear']:
            my_outputs['hz_U'] = self._tVariables['U_hz'+suffix]
            my_outputs['hz_L'] = self._tVariables['L_hz'+suffix]
            my_outputs['hz_W_alpha_U'] = self._tVariables['U_hz_W_alpha'+suffix]
            my_outputs['hz_W_alpha_L'] = self._tVariables['L_hz_W_alpha'+suffix]
            my_outputs['p_Z_embedded_U'] = self._tVariables['U_p_Z_embedded'+suffix]
            my_outputs['p_Z_embedded_L'] = self._tVariables['L_p_Z_embedded'+suffix]
            my_outputs['p_alpha_Z_input_W_U'] = self._tVariables['U_p_alpha_Z_input_W'+suffix]
            my_outputs['p_alpha_Z_input_W_L'] = self._tVariables['L_p_alpha_Z_input_W'+suffix]
        return my_outputs


    def _buildModel(self):
        """
                                       ******BUILD DiscreteSemiVAE GRAPH******
        """
        #Inputs to graph
        XU   = T.matrix('XU',   dtype=config.floatX)
        XL   = T.matrix('XL',   dtype=config.floatX)
        Y   = T.ivector('Y')
        epsU = T.matrix('epsU', dtype=config.floatX)
        epsL = T.matrix('epsL', dtype=config.floatX)
        self._fakeData(XU,XL,Y,epsU,epsL)
        self.updates_ack = True
        #Learning Rates and annealing objective function
        #Add them to npWeights/tWeights to be tracked [do not have a prefix _W or _b so wont be diff.]
        self._addWeights('lr', np.asarray(self.params['lr'],dtype=config.floatX),borrow=False)
        self._addWeights('annealKL', np.asarray(0.,dtype=config.floatX),borrow=False)
        self._addWeights('annealCW', np.asarray(0.,dtype=config.floatX),borrow=False)
        self._addWeights('update_ctr', np.asarray(1.,dtype=config.floatX),borrow=False)

        lr  = self.tWeights['lr']
        annealKL = self.tWeights['annealKL']
        annealCW = self.tWeights['annealCW']
        iteration_t    = self.tWeights['update_ctr']

        lr_update = [(lr,T.switch(lr/1.0005<1e-4,lr,lr/1.0005))]
        annealKL_div     = float(self.params['annealKL']) #50000.
        annealCW_div     = float(self.params['annealCW']) #50000.
        ctr_update = [(iteration_t, iteration_t+1)]
        annealKL_update  = [(annealKL,T.switch(iteration_t/annealKL_div>1,1.,0.01+iteration_t/annealKL_div))]
        annealCW_update  = [(annealCW,T.switch(iteration_t/annealCW_div>1,1.,0.01+iteration_t/annealCW_div))]

        Y_onehot = T.extra_ops.to_one_hot(Y,self.params['nclasses'],dtype=config.floatX)
        meanAbsDev = 0
        #Build training graphs
        graphprefixU = 'U'
        graphprefixL = 'L'
        graphprefixC = 'q(y|x)'
        #_,_,_,boundU_t,objfuncU_t=self._buildGraph(XU,epsU,self.params['betaprior'],
        outputsU_t = self._buildGraph(XU,epsU,self.params['betaprior'],
                                      Y=None,
                                      dropout_prob=self.params['input_dropout'],
                                      add_noise=True,
                                      annealKL=annealKL,
                                      evaluation=False,
                                      modifiedBatchNorm=False,
                                      graphprefix=graphprefixU)#use BN stats from U for L
        boundU_t = outputsU_t['bound']
        objfuncU_t = outputsU_t['objfunc']
        outputsL_t = self._buildGraph(XL,epsL,self.params['betaprior'],
                                      Y=Y_onehot,
                                      dropout_prob=self.params['input_dropout'],
                                      add_noise=True,
                                      annealKL=annealKL,
                                      evaluation=False,
                                      modifiedBatchNorm=self.params['modifiedBatchNorm'],
                                      graphprefix=graphprefixL)
        boundL_t = outputsL_t['bound']
        objfuncL_t = outputsL_t['objfunc']
        _,crossentropyloss_t,ncorrect_t = self._buildClassifier(XL, Y, 
                                      add_noise=True, 
                                      dropout_prob=self.params['input_dropout'], 
                                      evaluation=False, 
                                      modifiedBatchNorm=self.params['modifiedBatchNorm'], 
                                      graphprefix=graphprefixC)
        trainboundU = boundU_t.sum()
        trainboundL = boundL_t.sum()
        trainclassifier = self.params['classifier_weight']*crossentropyloss_t.sum()
        trainbound = trainboundU + trainboundL
        trainloss = trainbound + trainclassifier
        trainobjective = objfuncU_t.sum() + self.params['boundXY_weight']*objfuncL_t.sum() + annealCW*self.params['classifier_weight']*crossentropyloss_t.sum()
        trainobj_components = [objfuncU_t.sum(),objfuncL_t.sum(),self.params['classifier_weight']*crossentropyloss_t.sum()]

        #Build evaluation graph
        outputsU_e = self._buildGraph(XU,epsU,self.params['betaprior'],
                                      Y=None,
                                      dropout_prob=0.,
                                      add_noise=False,
                                      annealKL=None,
                                      evaluation=True,
                                      graphprefix=graphprefixU)
        boundU_e = outputsU_e['bound']
        objfuncU_e = outputsU_e['objfunc']
        outputsL_e = self._buildGraph(XL,epsL,self.params['betaprior'],
                                      Y=Y_onehot,
                                      dropout_prob=0.,
                                      add_noise=False,
                                      annealKL=None,
                                      evaluation=True,
                                      modifiedBatchNorm=self.params['modifiedBatchNorm'],
                                      graphprefix=graphprefixL)
        boundL_e = outputsL_e['bound']
        objfuncL_e = outputsL_e['objfunc']
        _,crossentropyloss_e,ncorrect_e = self._buildClassifier(XL, Y, 
                                      add_noise=False, 
                                      dropout_prob=0.,
                                      evaluation=True, 
                                      modifiedBatchNorm=self.params['modifiedBatchNorm'], 
                                      graphprefix=graphprefixC)
        evalboundU = boundU_e.sum()
        evalboundL = boundL_e.sum()
        evalbound = evalboundU + evalboundL
        evalclassifier = self.params['classifier_weight']*crossentropyloss_e.sum()
        evalloss = evalbound + evalclassifier
        evalobjective = objfuncU_e.sum() + objfuncL_e.sum() + self.params['classifier_weight']*crossentropyloss_e.sum()

        #Optimizer with specification for regularizer
        model_params = self._getModelParams()
        nparams = float(self._countParams(model_params))
        #setup grad norm (scale grad norm according to # parameters)
        if self.params['grad_norm'] == None:
            grad_norm_per_1000 = 1.0
        else:
            grad_norm_per_1000 = self.params['grad_norm']
        grad_norm = nparams/1000.0*grad_norm_per_1000
        self._p('# params to optimize = %s, max gradnorm = %s' % (nparams,grad_norm))


        if self.params['divide_grad']:
            divide_grad = T.cast(XU.shape[0],config.floatX)
        else:
            divide_grad = None
        optimizer_up, norm_list  = self._setupOptimizer(trainobjective, model_params,lr = lr,
                                                        reg_type =self.params['reg_type'],
                                                        reg_spec =self.params['reg_spec'],
                                                        reg_value= self.params['reg_value'],
                                                       grad_norm = grad_norm,
                                                       divide_grad = divide_grad)
        #self.updates is container for all updates (e.g. see _BNlayer in BaseModel)
        self.updates += optimizer_up+annealKL_update+annealCW_update+ctr_update

        #Build theano functions
        fxn_inputs = [XU,XL,Y,epsU,epsL]

        #Importance sampled estimate
#        ll_prior             = self._llGaussian(z_e, T.zeros_like(z_e,dtype=config.floatX),
#                                                    T.zeros_like(z_e,dtype=config.floatX))
#        ll_posterior         = self._llGaussian(z_e, mu_e, logcov_e)
#        ll_estimate          = -1*negCLL_e+ll_prior.sum(1,keepdims=True)-ll_posterior.sum(1,keepdims=True)
#        self.likelihood      = theano.function(fxn_inputs,ll_estimate)
        #outputs_train = [trainloss,ncorrect_t,trainbound,anneal.sum(),trainboundU,trainboundL,trainclassifier]
        outputs_train = {'cost':trainloss,
                         'ncorrect':ncorrect_t,
                         'bound':trainbound,
                         'annealKL':annealKL.sum(),
                         'annealCW':annealCW.sum(),
                         'boundU':trainboundU,
                         'boundL':trainboundL,
                         'classification_loss':trainclassifier,
                         }
        for k,v in self._getModelOutputs(outputsU_t,outputsL_t,suffix='_t').iteritems():
            outputs_train[k] = v

        outputs_eval = { 'cost':evalloss,
                         'ncorrect':ncorrect_e,
                         'bound':evalbound,
                         'boundU':evalboundU,
                         'boundL':evalboundL,
                         'classification_loss':evalclassifier,
                        }
        for k,v in self._getModelOutputs(outputsU_e,outputsL_e,suffix='_e').iteritems():
            outputs_eval[k] = v

        # add batchnorm running statistics to output
        for k,v in self.tWeights.iteritems():
            if 'running' in k:
                outputs_train[k] = v

        self.train      = theano.function(fxn_inputs, outputs_train,
                                              updates = self.updates, name = 'Train')
        outputs_debug = outputs_train #+ norm_list + trainobj_components
        self.debug      = theano.function(fxn_inputs, outputs_debug,
                                              updates = self.updates, name = 'Train+Debug')
#        self.inference  = theano.function(fxn_inputs, [z_e, mu_e, logcov_e], name = 'Inference')
        #outputs_eval = [evalloss,ncorrect_e,evalbound]
        self.evaluate   = theano.function(fxn_inputs, outputs_eval, name = 'Evaluate')
        self.decay_lr   = theano.function([],lr.sum(),name = 'Update LR',updates=lr_update)
        Z = T.matrix('Z',dtype=config.floatX)
        alpha = T.matrix('alpha',dtype=config.floatX)
        (pX_U,), _ = self._buildEmission(alpha=alpha,Z=Z,X=XU,
                                      graphprefix=graphprefixU,
                                      evaluation=True,
                                      modifiedBatchNorm=False)
        (pX_L,), _ = self._buildEmission(alpha=alpha,Z=Z,X=XL,
                                      graphprefix=graphprefixL,
                                      evaluation=True,
                                      modifiedBatchNorm=self.params['modifiedBatchNorm'])
        self.reconstructU= theano.function([Z,alpha], [pX_U], name='Reconstruct_U')
        self.reconstructL= theano.function([Z,alpha], [pX_L], name='Reconstruct_L')
#        self.reset_anneal=theano.function([],anneal.sum(), updates = [(anneal,0.01)], name='reset anneal')

    def sample(self,onehot=True,nsamples=100):
        """
                                Sample from Generative Model
        """
        K = self.params['nclasses']
        if onehot:
            z = np.random.randn(K*nsamples,self.params['dim_stochastic']).astype(config.floatX)
            alpha = np.repeat((np.arange(K).reshape(1,-1) == np.arange(K).reshape(-1,1)).astype('int'),axis=0,repeats=nsamples).astype(config.floatX)
            return {'U':self.reconstructU(z,alpha),'L':self.reconstructL(z,alpha)}
        else:
            z = np.random.randn(nsamples,self.params['dim_stochastic']).astype(config.floatX)
            u = np.random.randn(nsamples,K)
            alpha = np.exp(u-scipy.misc.logsumexp(u,axis=1,keepdims=True)).astype(config.floatX)
            return {'U':self.reconstructU(z,alpha),'L':self.reconstructL(z,alpha)}
