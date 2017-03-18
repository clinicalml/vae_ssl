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

class GaussianMixtureSemiVAE(LogGammaSemiVAE):

    def __init__(self,params,paramFile=None,reloadFile=None):
        super(GaussianMixtureSemiVAE,self).__init__(params,paramFile=paramFile,reloadFile=reloadFile)

    def _createParams(self):
        """
                    _createParams: create parameters necessary for the model
        """
        self.track_params=[]
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
        for qy_l in range(self.params['y_inference_layers']):
            dim_input     = 2*self.params['dim_alpha'] 
            dim_output    = 2*self.params['dim_alpha'] 
            if self.params['nonlinearity']=='maxout':
                dim_output *= self.params['maxout_stride']
            # q(y|x)
            npWeights['q_y_x_'+str(qy_l)+'_W'] = self._getWeight((dim_input, dim_output))
            npWeights['q_y_x_'+str(qy_l)+'_b'] = self._getWeight((dim_output, ))
        for hz_l in range(self.params['hz_inference_layers']):
            dim_input = DIM_HIDDEN
            dim_output = DIM_HIDDEN
            if self.params['nonlinearity']=='maxout':
                dim_output = DIM_HIDDEN*self.params['maxout_stride']
            npWeights['q_hz_%s_W' % hz_l] = self._getWeight((dim_input, dim_output))
            npWeights['q_hz_%s_b' % hz_l] = self._getWeight((dim_output, ))
        for z_l in range(self.params['z_inference_layers']):
            dim_input     = DIM_HIDDEN
            if z_l == 0: 
                dim_input = self.params['dim_alpha']+DIM_HIDDEN
            dim_output    = DIM_HIDDEN
            if self.params['nonlinearity']=='maxout':
                dim_output= DIM_HIDDEN*self.params['maxout_stride']
            # Z
            npWeights['q_Z_'+str(z_l)+'_W'] = self._getWeight((dim_input, dim_output))
            npWeights['q_Z_'+str(z_l)+'_b'] = self._getWeight((dim_output, ))

        if self.params['inference_model']=='single':
            npWeights['q_hz2_mu_W'] = self._getWeight((DIM_HIDDEN, self.params['dim_alpha']))
            npWeights['q_hz2_mu_b'] = self._getWeight((self.params['dim_alpha'],))
            npWeights['q_hz2_logcov2_W'] = self._getWeight((DIM_HIDDEN, self.params['dim_alpha']))
            npWeights['q_hz2_logcov2_b'] = self._getWeight((self.params['dim_alpha'],))
            dim_output = DIM_HIDDEN
            if self.params['nonlinearity']=='maxout':
                dim_output= DIM_HIDDEN*self.params['maxout_stride']
            npWeights['q_y_x_W'] = self._getWeight((2*self.params['dim_alpha'],self.params['nclasses']))
            npWeights['q_y_x_b'] = self._getWeight((self.params['nclasses'],))
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
        dim_input += self.params['dim_alpha']
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
        K = self.params['nclasses']
        for y in range(K):
            npWeights['p_alpha|y=%s_mu_W'%y] = self._getWeight((self.params['dim_alpha'],))
            npWeights['p_alpha|y=%s_logcov2_W'%y] = self._getWeight((self.params['dim_alpha'],))
            self.track_params += ['p_alpha|y=%s_mu_W'%y, 'p_alpha|y=%s_logcov2_W'%y]
        return npWeights

    def _KL_Gaussian(self, mu_q, logcov2_q, mu_p, logcov2_p):
        """
            KL-divergence: int_z q(z)log(q(z)/p(z)) dz
        """
        mq = mu_q
        mp = mu_p
        sq = logcov2_q
        sp = logcov2_p 
        return T.sum(-0.5*(sq-sp)-0.5*(1-T.exp(sq-sp)-T.exp(sp)*((mq-mp)**2.0)),axis=1)
        
    def _variationalGaussianMixture(self, probs, mu, logcov2, eps):
        alpha = mu + T.exp(0.5*logcov2)*eps

        K = self.params['nclasses']
        logprobs = T.log(probs)
        KL_Y = T.sum(probs*(logprobs-T.log(1./K)),axis=1)

        KL_alpha = []
        for y in range(K):
            mu_p = self.tWeights['p_alpha|y=%s_mu_W'%y].reshape((1,-1))
            logcov2_p = self.tWeights['p_alpha|y=%s_logcov2_W'%y].reshape((1,-1))
            KL_alpha.append(self._KL_Gaussian(mu,logcov2,mu_p,logcov2_p).reshape((-1,1)))
        KL_alpha = T.concatenate(KL_alpha,axis=1)
        KL_alpha = T.sum(probs*KL_alpha,axis=1)
        return alpha, KL_alpha+KL_Y

    def _variationalGaussianConditional(self, Y, mu_q, logcov2_q, eps_q):
        alpha = mu_q + T.exp(0.5*logcov2_q)*eps_q

        K = self.params['nclasses']
        mu_p = []
        logcov2_p = []
        for y in range(K):
            mu_p.append(self.tWeights['p_alpha|y=%s_mu_W'%y].reshape((1,-1)))
            logcov2_p.append(self.tWeights['p_alpha|y=%s_logcov2_W'%y].reshape((1,-1)))
        mu_p = T.concatenate(mu_p,axis=0)
        logcov2_p = T.concatenate(logcov2_p,axis=0)
        mu_p = T.dot(Y,mu_p)
        logcov2_p = T.dot(Y,logcov2_p)
        KL_alpha = self._KL_Gaussian(mu_q,logcov2_q,mu_p,logcov2_p)
        return alpha, KL_alpha

    def _reconstruct(self, X, Y, Z, evaluation=True, graphprefix=None):
        Y = T.extra_ops.to_one_hot(Y,self.params['nclasses'],dtype=config.floatX)
        K = self.params['nclasses']
        mu_p = []
        logcov2_p = []
        for y in range(K):
            mu_p.append(self.tWeights['p_alpha|y=%s_mu_W'%y].reshape((1,-1)))
            logcov2_p.append(self.tWeights['p_alpha|y=%s_logcov2_W'%y].reshape((1,-1)))
        mu_p = T.concatenate(mu_p,axis=0)
        logcov2_p = T.concatenate(logcov2_p,axis=0)
        mu_p = T.dot(Y,mu_p)
        logcov2_p = T.dot(Y,logcov2_p)
        eps = self.srng.normal(mu_p.shape,dtype=config.floatX)
        alpha = mu_p + T.exp(0.5*logcov2_p)*eps
        (pX,), _ = self._buildEmission(alpha=alpha,Z=Z,X=X,
                                      graphprefix=graphprefix,
                                      evaluation=True
                                      )
        return pX

        
    def _buildRecognitionBase(self, X, dropout_prob=0.,evaluation=False,graphprefix=None):
        if self.params['separateBNrunningstats']:
            self._BNprefix=graphprefix
        else:
            self._BNprefix=None
        self._p(('Inference with dropout :%.4f')%(dropout_prob))
        inp = self._dropout(X,dropout_prob)
        hx = self._buildHiddenLayers(inp,self.params['q_layers'],'q_h(x)_{layer}',evaluation)
        return hx

    def _buildInferenceAlpha(self, hx, dropout_prob=0.,evaluation=False,graphprefix=None):
        if self.params['separateBNrunningstats']:
            self._BNprefix=graphprefix
        else:
            self._BNprefix=None
        if dropout_prob > 0:
            self._p(('Inference with dropout :%.4f')%(dropout_prob))
            hx = self._dropout(hx,dropout_prob)
        hz2 = self._buildHiddenLayers(hx,self.params['alpha_inference_layers'],'q_hz2_{layer}',evaluation)
        return hz2

    def _buildInferenceY(self, hz2, dropout_prob=0.,evaluation=False,graphprefix=None):
        if self.params['separateBNrunningstats']:
            self._BNprefix=graphprefix
        else:
            self._BNprefix=None
        if dropout_prob > 0:
            self._p(('Inference with dropout :%.4f')%(dropout_prob))
            hz2 = self._dropout(hz2,dropout_prob)
        hy = self._buildHiddenLayers(hz2,self.params['y_inference_layers'],'q_y_x_{layer}',evaluation)
        return hy

    def _build_qz(self,alpha,hx,evaluation,graphprefix):
        if self.params['separateBNrunningstats']:
            self._BNprefix=graphprefix
        else:
            self._BNprefix=None
        if self.params['dropout_hx']>0 and evaluation==False:
            hx = self._dropout(hx,self.params['dropout_hx'])
        hz = self._buildHiddenLayers(hx,self.params['hz_inference_layers'],'q_hz_{layer}',evaluation)
        suffix = '_e' if evaluation else '_t'
        #merge h(x) and alpha
        hz_alpha = T.concatenate([alpha,hz],axis=1)
        #infer mu and logcov
        q_Z_hidden = self._buildHiddenLayers(hz_alpha,self.params['z_inference_layers'],'q_Z_{layer}',evaluation)
        mu      = self._LinearNL(self.tWeights['q_Z_mu_W'],self.tWeights['q_Z_mu_b'],q_Z_hidden,onlyLinear=True)
        logcov  = self._LinearNL(self.tWeights['q_Z_logcov_W'],self.tWeights['q_Z_logcov_b'],q_Z_hidden,onlyLinear=True)
        return mu, logcov

    def _buildGraph(self, X, eps, Y=None, dropout_prob=0.,add_noise=False,annealKL=None,evaluation=False,graphprefix=None):
        """
                                Build VAE subgraph to do inference and emissions
                (if Y==None, build upper bound of -logp(x), else build upper bound of -logp(x,y)
        """
        if Y == None:
            self._p(('Building graph for lower bound of logp(x)'))
        else:
            self._p(('Building graph for lower bound of logp(x,y)'))

        suffix = '_e' if evaluation else '_t'

        hx = self._buildRecognitionBase(X,dropout_prob,evaluation,graphprefix)
        self._addVariable(graphprefix+'_h(x)'+suffix,hx,ignore_warnings=True)

        hz2 = self._buildInferenceAlpha(hx,0.,evaluation,graphprefix)
        self._addVariable(graphprefix+'_hz2'+suffix,hx,ignore_warnings=True)

        mu_z2 = self._LinearNL(W=self.tWeights['q_hz2_mu_W'],b=self.tWeights['q_hz2_mu_b'],inp=hz2,onlyLinear=True)
        logcov2_z2 = self._LinearNL(W=self.tWeights['q_hz2_logcov2_W'],b=self.tWeights['q_hz2_logcov2_b'],inp=hz2,onlyLinear=True)
        self._addVariable(graphprefix+'_Z2_mu'+suffix,mu_z2,ignore_warnings=True)
        self._addVariable(graphprefix+'_Z2_logcov2'+suffix,logcov2_z2,ignore_warnings=True)

        hy = self._buildInferenceY(T.concatenate([mu_z2,logcov2_z2],axis=1),0.,evaluation,graphprefix)
        qy_x = T.nnet.softmax(self._LinearNL(W=self.tWeights['q_y_x_W'],b=self.tWeights['q_y_x_b'],inp=hy,onlyLinear=True))

        eps2 = self.srng.normal(mu_z2.shape,dtype=config.floatX)
        if Y is None:
            alpha, KL_alpha = self._variationalGaussianMixture(qy_x,mu_z2,logcov2_z2,eps2)
        else:
            alpha, KL_alpha = self._variationalGaussianConditional(Y,mu_z2,logcov2_z2,eps2)
            batchsize = alpha.shape[0]
            K = self.params['nclasses']
            nllY = batchsize*T.log(K)
        if add_noise:
            alpha = alpha + self.srng.normal(alpha.shape,0,0.05,dtype=config.floatX)

        mu, logcov = self._build_qz(alpha,hx,evaluation,graphprefix)
        self._addVariable(graphprefix+'_mu'+suffix,mu,ignore_warnings=True)
        self._addVariable(graphprefix+'_logcov2'+suffix,logcov,ignore_warnings=True)
        #Z = gaussian variates
        Z, KL_Z = self._variationalGaussian(mu,logcov,eps)
        if add_noise:
            Z = Z + self.srng.normal(Z.shape,0,0.05,dtype=config.floatX)
        #generative model
        _, nllX = self._buildEmission(alpha, Z, X, graphprefix, evaluation=evaluation)

        #negative of the lower bound
        KL = KL_alpha.sum() + KL_Z.sum()
        NLL = nllX.sum()
        if Y!=None:
            NLL += nllY.sum()
        bound = KL + NLL
        #objective function
        if annealKL is None:
            objfunc = bound
        else:
            objfunc = annealKL*KL + NLL

        outputs = {'alpha':alpha,
                   'Z':Z,
                   'bound':bound,
                   'objfunc':objfunc,
                   'nllX':nllX,
                   'KL_alpha':KL_alpha,
                   'KL_Z':KL_Z,
                   'KL':KL,
                   'NLL':NLL,
                   'probs':qy_x,
                   }
        if Y!=None:
            outputs['nllY']=nllY
            outputs['Y']=Y

        return outputs

    def _buildClassifier(self,probs,Y):
        loss= T.nnet.categorical_crossentropy(probs,Y)
        ncorrect = T.eq(T.argmax(probs,axis=1),Y).sum()
        return probs, loss, ncorrect

    def _getModelOutputs(self,outputsU,outputsL,suffix=''):
        my_outputs = {
                         'KL_alpha_U':outputsU['KL_alpha'].sum(),
                         'nllX_U':outputsU['nllX'].sum(),
                         'NLL_U':outputsU['NLL'].sum(),
                         'KL_U':outputsU['KL'].sum(),
                         'KL_Z_U':outputsU['KL_Z'].sum(),
                         'nllY':outputsL['nllY'].sum(),
                         'KL_alpha_L':outputsL['KL_alpha'].sum(),
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
                         'probs_U':outputsU['probs'],
                         'probs_L':outputsL['probs']
                         }
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
        if 'annealKL' not in self.params:
            self.params['annealKL'] = self.params['annealKL_alpha']

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
        outputsU_t = self._buildGraph(XU,epsU,
                                      Y=None,
                                      dropout_prob=self.params['input_dropout'],
                                      add_noise=True,
                                      annealKL=annealKL,
                                      evaluation=False,
                                      graphprefix=graphprefixU)#use BN stats from U for L
        boundU_t = outputsU_t['bound']
        objfuncU_t = outputsU_t['objfunc']
        outputsL_t = self._buildGraph(XL,epsL,
                                      Y=Y_onehot,
                                      dropout_prob=self.params['input_dropout'],
                                      add_noise=True,
                                      annealKL=annealKL,
                                      evaluation=False,
                                      graphprefix=graphprefixL)
        boundL_t = outputsL_t['bound']
        objfuncL_t = outputsL_t['objfunc']
        probsL_t = outputsL_t['probs']
        _,crossentropyloss_t,ncorrect_t = self._buildClassifier(probsL_t, Y) 
        trainboundU = boundU_t.sum()
        trainboundL = boundL_t.sum()
        trainclassifier = self.params['classifier_weight']*crossentropyloss_t.sum()
        trainbound = trainboundU + trainboundL
        trainloss = trainbound + trainclassifier
        trainobjective = objfuncU_t.sum() + self.params['boundXY_weight']*objfuncL_t.sum() + annealCW*self.params['classifier_weight']*crossentropyloss_t.sum()
        trainobj_components = [objfuncU_t.sum(),objfuncL_t.sum(),self.params['classifier_weight']*crossentropyloss_t.sum()]

        #Build evaluation graph
        outputsU_e = self._buildGraph(XU,epsU,
                                      Y=None,
                                      dropout_prob=0.,
                                      add_noise=False,
                                      annealKL=None,
                                      evaluation=True,
                                      graphprefix=graphprefixU)
        boundU_e = outputsU_e['bound']
        objfuncU_e = outputsU_e['objfunc']
        outputsL_e = self._buildGraph(XL,epsL,
                                      Y=Y_onehot,
                                      dropout_prob=0.,
                                      add_noise=False,
                                      annealKL=None,
                                      evaluation=True,
                                      graphprefix=graphprefixL)
        boundL_e = outputsL_e['bound']
        objfuncL_e = outputsL_e['objfunc']
        probsL_e = outputsL_e['probs']
        _,crossentropyloss_e,ncorrect_e = self._buildClassifier(probsL_e, Y) 
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
                         'annealKL_alpha':annealKL.sum(),
                         'annealKL_Z':annealKL.sum(),
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
                         'Y':Y_onehot,
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
        YU = T.ivector('YU')
        YL = T.ivector('YL')
        pX_U = self._reconstruct(XU,YU,Z,graphprefix=graphprefixU)
        pX_L = self._reconstruct(XL,YL,Z,graphprefix=graphprefixL)
        self.reconstructU= theano.function([YU,Z], [pX_U], name='Reconstruct_U')
        self.reconstructL= theano.function([YL,Z], [pX_L], name='Reconstruct_L')
#        self.reset_anneal=theano.function([],anneal.sum(), updates = [(anneal,0.01)], name='reset anneal')

    def sample(self,onehot=True,nsamples=100):
        """
                                Sample from Generative Model
        """
        K = self.params['nclasses']
        if onehot:
            z = np.random.randn(K*nsamples,self.params['dim_stochastic']).astype(config.floatX)
            y = np.repeat(np.arange(K),repeats=nsamples).astype('int32')
            #y = np.repeat((np.arange(K).reshape(1,-1) == np.arange(K).reshape(-1,1)).astype('int'),axis=0,repeats=nsamples).astype(config.floatX)
            return {'U':self.reconstructU(y,z),'L':self.reconstructL(y,z)}
        else:
            return {'U':np.zeros(1),'L':np.zeros(1)}
