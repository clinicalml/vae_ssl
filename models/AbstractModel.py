import six.moves.cPickle as pickle
import sys, time, os
import numpy as np
import gzip, warnings
import theano
from theano import config
theano.config.compute_test_value = 'warn'
import theano.tensor as T
from theanomodels.utils.optimizer import adam,rmsprop
from theanomodels.utils.misc import saveHDF5
from theanomodels.models import BaseModel
from contextlib import contextmanager
from contextlib import nested 
from nestd import NestD
from nestdarrays import NestDArrays
import json
import optimizer
from dataloader import DataLoader
from progressbar import ProgressBar
import gc

IGNORE_WARNINGS=True


class AbstractModel(BaseModel, object):
    _evaluating = False

    def _createParams(self):
        return {}

    def __init__(self, params, configFile, reloadDir=None, logfile=None):
        if logfile is not None:
            assert isinstance(logfile, file), 'if logfile is not None, it must be a type of file'
        self.logfile = logfile
        
        with open(configFile,'w') as f:
            f.write(json.dumps(params,sort_keys=True,indent=2,separators=(',',':')))
        if configFile[-5:]=='.json':
            paramFile = configFile[:-5]+'.pkl'
        else:
            paramFile = configFile+'.pkl'
        super(AbstractModel,self).__init__(params,
                                           paramFile=paramFile,
                                           reloadFile=reloadDir)

    def get_nestd(self,attr_name):
        if not hasattr(self,attr_name):
            setattr(self,attr_name,NestD())
        attr = getattr(self,attr_name)
        assert isinstance(attr,dict), '%s must be a type of dict' % attr_name
        return attr

    @contextmanager
    def namespace(self,key,attr_name='tWeights'):
        """
        * sets self.__dict__[attr_name] to self.__dict__[attr_name][key]
        * if attr_name does not exist, then it is created
        * if key does not exist in attr_name, then it is created
        example usage:
        ```
        self.tWeights['layer_weights'] = {'w':np.random.randn(5)}
        with self.attr_name('layer_weights','tWeights'):
            w = self.tWeights['w']

        self.tWeights['layers'] = {'layer1':{'w':np.random.randn(5)}}
        with self.attr_name('layers','tWeights'), self.attr_name('layer1','tWeights'):
            w = self.tWeights['w']
        ```
        """
        #do this stuff before executing stuff under the with statement:
        attr = self.get_nestd(attr_name)
        if key not in attr:
            attr[key] = NestD()
        temp = attr
        setattr(self,attr_name,attr[key])
        yield #wait until we finish executing the with statement"
        #now that we've exited with statement, do this stuff:
        setattr(self,attr_name,temp)

    default_namespaces = ['tWeights','tOutputs']

    @contextmanager
    def namespaces(self,keys,attr_names=None):
        """
        converts attr_names and keys to lists, if they are not already lists
        for a in attr_names:
            for k in keys:
                set the self.namespace(a,k)

        if attr_names is None, then attr_names=self.default_namespaces

        example usage:
        ```
        self.attr1 = {}
        self.attr2 = {}
        with self.namespaces(['key1','key2'],['attr1','attr2']):
            self.attr1['A'] = 1
            self.attr2['B'] = 2
        print self.attr1
        >> {'key1':{'key2':{'A':1}}}
        print self.attr2
        >> {'key1':{'key2':{'B':2}}}
        ```
        """
        if attr_names is None:
            attr_names = self.default_namespaces
        if not isinstance(attr_names,list):
            attr_names=[attr_names]
        if not isinstance(keys,list):
            keys =[keys]
        assert len(attr_names) > 0, 'len(attr_names) cannot be zero'
        assert len(keys) > 0, 'len(keys) cannot be zero'
        
        managers = []
        for a in attr_names:
            for k in keys:
                managers.append(self.namespace(k,a))
        with nested(*managers):
            yield

    def print_namespace(self,attr_name):
        assert hasattr(self,attr_name),'self does not have attribute %s' % attr_name
        attr = self.get_nestd(attr_name)
        assert isinstance(attr,NestD),'%s must be a type of NestD)' % attr
        for k,v in sorted(attr.walk()):
            path = '/'.join(k)
            print_str = path + ':  '
            if hasattr(v,'get_value'):
                print_str += 'shared: %s%s' % (v.type,v.get_value().shape)
            elif hasattr(v,'tag') and hasattr(v.tag,'test_value'):
                v = v.tag.test_value
                print_str += 'tag.test_value: %s%s' % (v.dtype,v.shape)
            else:
                print_str += '%s' % v
            print print_str

    @contextmanager
    def evaluate(self):
        """
        sets self._evaluating=True

        example usage:
        ```
        with self.evaluate():
            #do stuff
        ```
        """
        if not hasattr(self,'_evaluating'):
            self._evaluating=False
        self._evaluating=True
        yield
        self._evaluating=False

    @contextmanager
    def set_attr(self,attr_name,value):
        """
        sets attr_name to value
        """
        if not hasattr(self,attr_name):
            setattr(self,attr_name,None)
        temp = getattr(self,attr_name)
        setattr(self,attr_name,value)
        yield
        setattr(self,attr_name,temp)

    def count_params(self,params=None):
        """
        count_params: Count the number of parameters in the model that will be optimized
        """
        if params==None:
            self.nParams    = 0
            for k in self.npWeights:
                ctr = np.array(self.npWeights[k].shape).prod()
                self.nParams+= ctr
            self._p(('Nparameters: %d')%self.nParams)
            return self.nParams
        else:
            nParams    = 0
            for p in params:
                ctr = np.array(p.get_value().shape).prod()
                nParams+= ctr
            return nParams
            

    def add_shared(self,name,data,namespace_name,ignore_warnings=IGNORE_WARNINGS,**kwargs):
        """
        Add theano shared to namespace 
        
        name in theano.shared will be '/'.join(namespace_path('tWeights')+[name])
        e.g. if current namespace is tWeights['weights']['layer1'] and weight has name 'W'
        the fullpath will be 'weights/layer1/W'
        """
        namespace = getattr(self,namespace_name)
        if name not in namespace:
            fullpath = '/'.join([str(s) for s in namespace.path()]+[name])
            namespace[name]  = theano.shared(data.astype(config.floatX),name=fullpath,**kwargs)
        else:
            if not ignore_warnings:
                warnings.warn(name+" found in tWeights. No action taken")
        return namespace[name]

    def add_weights(self, name, data, **kwargs):
        """
        Add to tWeights (under current namespace)
        
        name in theano.shared will be '/'.join(namespace_path('tWeights')+[name])
        e.g. if current namespace is tWeights['weights']['layer1'] and weight has name 'W'
        the fullpath will be 'weights/layer1/W'
        """
        return self.add_shared(name,data,'tWeights',**kwargs)

    def add_update(self, var, data, ignore_warnings=False):
        """
        Add an update for tWeights
        """
        if len(self.updates) > 0 and var in zip(*self.updates)[0]:
            if not ignore_warnings:
                warnings.warn(var.name+' found in self.updates...no action taken')
        else:
            self.updates.append((var,data))

    def get_model_params(self, restrict = ''):
        """
        Return list of model parameters to take derivatives with respect to
        """
        paramlist = []
        namelist  = []
        otherparamnames = []
        for k in self.tWeights.values():
            if 'W_' in k.name or 'b_' in k.name or '_b' in k.name or '_W' in k.name or 'U_' in k.name or '_U' in k.name:
                #Use this to only get a list of parameters with specific substrings like 'p_'
                #Since it is set to '' by default, it should appear in all strings
                if restrict in k.name:
                    paramlist.append(k)
                    namelist.append(k.name)
        othernames = [k.name for k in self.tWeights.values() if k.name not in namelist]

        self._p('Params to optimize:\n' + '\n'.join(namelist))
        self._p('Other params:\n' + '\n'.join(othernames))
        return paramlist

    def dropout(self, X, p=0.):
        """
        dropout : X is the input, p is the dropout probability
        Do not need to do anything in the case of no dropout since we divide by retain prob.
        """
        if p > 0:
            retain_prob = 1 - p
            X *= self.srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
            X /= retain_prob
        return X
    
    def batchnorm(self, x, dimoutput, output_axis, ndims, bias=True, momentum=0.95, eps=1e-3, **kwargs):
        """
        Batch normalization layer
        https://arxiv.org/abs/1502.03167
        * x: input to batch normalization
        * dimoutput: size of output_axis dimensions
        * output_axis: a separate set of batchnorm stats will be generated for each element in output_axis (e.g. following linear layers, output_axis=1; following convolutional layers, output_axis should be the channels dimension)
        """
        assert isinstance(output_axis,int), 'output_axis must be an int'
        assert isinstance(dimoutput,int), 'dimoutput must be an int'

        init_shape = (dimoutput,)
        gamma = self.add_weights('bn_gamma',self._getWeight(init_shape,**kwargs))
        
        running_mean = self.add_shared('bn_running_mean',np.zeros(init_shape),'tBatchnormStats')
        running_var = self.add_shared('bn_running_var',np.ones(init_shape),'tBatchnormStats')
        mom = self.add_shared('bn_momentum',np.asarray(0),'tBatchnormStats')

        #gamma = gamma.dimshuffle(bn_shape)

        #set of axes we will calculate batch norm statistics over
        axis = [i for i in range(ndims) if i!=output_axis]
        #bn_shape is the input to dimshuffle
        bn_shape = [0 if i==output_axis else 'x' for i in range(ndims)]
        _gamma = gamma.dimshuffle(bn_shape)
        _running_mean = running_mean.dimshuffle(bn_shape)
        _running_var = running_var.dimshuffle(bn_shape)

        if self._evaluating:
            y = (x-_running_mean)/T.sqrt(_running_var+eps)
        else:
            batch_mean = x.mean(axis,keepdims=True)
            batch_var = x.var(axis,keepdims=True)
            y = (x-batch_mean)/T.sqrt(batch_var+eps)
            
            batch_mean = batch_mean.squeeze()
            batch_var = batch_var.squeeze()

            #Update running stats
            m = T.cast(x.shape[0],config.floatX)
            self.add_update(running_mean, mom*running_mean+(1.-mom)*batch_mean)
            self.add_update(running_var, mom*running_var+(1.-mom)*batch_var*m/(m-1))
            #momentum will be 0 in the first iteration, and momentum in all subsequent iters
            self.add_update(mom,momentum)

        z = _gamma*y
        if bias:
            beta = self.add_weights('bn_beta',self._getWeight(init_shape,**kwargs))
            _beta = beta.dimshuffle(bn_shape)
            z = z+_beta

        return z
        
    def linear(self, x, diminput, dimoutput, bias=True, **kwargs):
        """
        * return T.dot(x,W)+b 
        * set bias=False to remove bias
        """
        W = self.add_weights('W',self._getWeight((diminput,dimoutput),**kwargs))
        y = T.dot(x,W)
        if bias:
            b = self.add_weights('b',self._getWeight((dimoutput,)))
            y = y + b
    
        #If only doing a dot product return as is
        return y
        
    def bilinear(self,x,y,dimx,dimy,dimoutput,bias=True,**kwargs):
        """
        return xTWy+bW should have shape (output_dim, x.shape[1], y.shape[1])
        """
        W = self.add_weights('W',self._getWeight((dimoutput,dimx,dimy),**kwargs))
        xW = T.dot(x,W)
        xWy = T.sum(xW*y.reshape((y.shape[0],1,-1)),axis=2)
        if bias:
            b = self.add_weights('b',self._getWeight((dimoutput,),**kwargs))
            xWy = xWy+b
        return xWy

    
    def variational_gaussian(self, mu, logcov, eps):
        """
                            KL divergence between N(0,I) and N(mu,exp(logcov))
        """
        #Pass z back
        z = mu + T.exp(0.5*logcov)*eps
        KL = 0.5*T.sum(-logcov -1 + T.exp(logcov) +mu**2 ,axis=1,keepdims=True)
        return z,KL

    def nll_gaussian(self,X,mu,logcov2):
        return 0.5*(np.log(2*np.pi)+logcov2+((X-mu)/T.exp(0.5*logcov2))**2)

    def nll_bernoulli(self,X,p):
        return T.nnet.binary_crossentropy(p,X)
        

    def setup_optimizer(self,objective,namespace,lr,optim_method='adam',
                        reg_value=0,reg_type='l2',
                        divide_grad=True,grad_norm=None,**kwargs):
        """
        setup_optimizer :   Wrapper for calling optimizer specified for the model. Internally also updates
                            the list of shared optimization variables in the model
        Calls self.optimizer to minimize "cost" wrt "params" using learning rate "lr", the other arguments are passed
        as is to the optimizer

        returns: updates (list of tuples specifying updates for all the shared variables in the model)
                 norm_list (for debugging, [0] : norm of parameters, [1] : norm of gradients, [2] : norm of optimization weights)
        """
        if optim_method=='adam':
            optim_method = optimizer.adam
        elif optim_method=='rmsprop':
            optim_method = optimizer.rmsprop
        else:
            assert False,'Invalid optimizer'

        # namespace.leaves() will yield a depth first iterator of namespace
        weights = namespace.leaves(sort_keypaths=True)

        #count number of weights for gradient normalization
        nparams = float(self.count_params(weights))

        #Add regularization
        if reg_value is not None and reg_value > 0:
            objective = optimizer.regularize(objective,weights,reg_value,reg_type)

        #gradients
        grads = T.grad(objective,weights)

        #normalize gradient according to mini-batch size
        if divide_grad:
            divide_grad = T.cast(XU.shape[0],config.floatX)
            grads = optimizer.rescale(grads,divide_grad)

        #setup grad norm (scale grad norm according to # parameters)
        if grad_norm is not None:
            grad_norm_per_1000 = self.params['grad_norm']
            grad_norm = nparams/1000.0*grad_norm_per_1000
            grads = optimizer.normalize(grads, grad_norm)

        self._p('# params to optimize = %s, max gradnorm = %s' % (nparams,grad_norm))

        outputs = optim_method(weights,grads,lr=lr,opt_params=self.tOptWeights,**kwargs)
        optimizer_up, norm_list, opt_params = outputs
 
        #If we passed in None initially then set optWeights
        if self.tOptWeights is None:
            self.tOptWeights = opt_params
        return optimizer_up, norm_list, objective

    def build_hidden_layers(self, h, diminput, dimoutput, nlayers, normalization=True, **kwargs):
        """
        Convenience function to build hidden layers
        """
        if self.params['nonlinearity']=='maxout':
            window = self.params['maxout_stride']
        else:
            window = 1
        for l in range(nlayers):
            with self.namespaces('layer'+str(l)):
                h = self.linear(h,diminput,window*dimoutput,**kwargs)
                y = h
                if self.params['batchnorm'] and normalization:
                    h = self.batchnorm(h,window*dimoutput,output_axis=1,ndims=2,**kwargs)
                h = self._applyNL(h)
                diminput = dimoutput
        return h
    
    def preprocess_minibatch(self,minibatch):
        return minibatch

    def run_epoch(self,dataset,runfunc,maxiters=None,collect_garbage=False):
        start_time = time.time()
        epoch_outputs = NestDArrays(axis=0,expand_dim=None)
        nbatches = len(dataset)
        with ProgressBar(nbatches) as pb:
            # note that the ProgressBar will temporarily
            # stop sending print outputs to any logfile attached to stdout, 
            # because of the  carriage returns it uses, which we don't want 
            # or need in our logfile, because it would make the logfile look 
            # funny and hard to read
            # also note that at the end of the with statement, we will write
            # the final output of the ProgressBar to the logfile, if one exists
            for i,data in enumerate(dataset):
                if collect_garbage:
                    gc.collect()
                if maxiters is not None and i >= maxiters:
                    break
                minibatch = self.preprocess_minibatch(data)
                # minibatch is assumed to be a dict
                batch_outputs = runfunc(**minibatch)
                # some of these are CudaNDArray, so convert to NDarray
                batch_outputs = NestD(batch_outputs).apply(np.asarray)
                epoch_outputs.append(batch_outputs)
                pb.update(i+1,self.progress_bar_update(epoch_outputs))
            # if logfile exists, write output to it
            if self.logfile:
                self.logfile.write(pb.get_current_output_str())
        duration = time.time() - start_time
        epoch_outputs.append({'duration (seconds)':duration})
        return epoch_outputs

    def progress_bar_report_map(self):
        # see self.progress_bar_update for use
        # use list to preserve order
        return [
            ('loss',np.mean,'%0.2f (epoch mean)'),
            ('accuracy',np.mean,'%0.2f (epoch mean)'),
        ]

    def progress_bar_update(self,epoch_outputs={}):
        # use list to preserve order
        report_map = self.progress_bar_report_map()
        report = []
        for k,f,s in report_map:
            if k in epoch_outputs:
                report.append(k+': '+s % f(epoch_outputs[k]))
        if len(report)>0:
            return report
        else:
            return None

    def post_train_hook(self,**kwargs):
        pass

    def post_valid_hook(self,**kwargs):
        pass

    def post_test_hook(self,**kwargs):
        pass

    def post_save_hook(self,**kwargs):
        pass

    def learn(self, dataset, epoch_start=0, epoch_end=1000, batchsize=200,
              savedir=None, savefreq=None, evalfreq=None,
              maxiters=None, collect_garbage=False): 

        traindata = DataLoader(dataset.train,batchsize,shuffle=True)
        validdata = DataLoader(dataset.valid,batchsize,shuffle=False)

        log = NestDArrays({'train':{},'valid':{},'test':{}})    
        log_verbose = NestDArrays({'train':{},'valid':{},'test':{}}) 

        for epoch in range(epoch_start,epoch_end+1):
            #train
            print '\nTraining: epoch %s of %s' % (epoch,epoch_end)
            epoch_log = self.run_epoch(traindata,self.train,maxiters,collect_garbage)
            self.post_train_hook(**{k:v for k,v in locals().iteritems() if k!='self'})
            log['train'].append(epoch_log.apply(np.mean))
            log['train'].append({'epoch':epoch})
            # log_verbose stores the last 100 training samples from each epoch
            log_verbose['train'].append(epoch_log.apply(lambda x: x[-100:]))

            if evalfreq is not None and epoch % evalfreq==0:
                #evaluate on validation set
                print '\nValidating: epoch %s of %s' % (epoch,epoch_end)
                epoch_log = self.run_epoch(validdata,self.evaluate,maxiters,collect_garbage)
                self.post_valid_hook(**{k:v for k,v in locals().iteritems() if k!='self'})
                log['valid'].append(epoch_log.apply(np.mean))
                log['valid'].append({'epoch':epoch})
                log_verbose['valid'].append(epoch_log.apply(lambda x: x[-100:]))

            if hasattr(dataset,'test') and epoch == epoch_end:
                #evaluate on test set
                print '\nTesting: epoch %s of %s' % (epoch,epoch_end)
                testdata = DataLoader(dataset.test,batchsize,shuffle=False)
                epoch_log = self.run_epoch(testdata,self.evaluate,maxiters,collect_garbage)
                self.post_test_hook(**{k:v for k,v in locals().iteritems() if k!='self'})
                log['test'].append(epoch_log.apply(np.mean))
                log['test'].append({'epoch':epoch})
                log_verbose['test'].append(epoch_log.apply(lambda x: x[-100:]))
            
            if savefreq is not None and (epoch % savefreq==0 or epoch == epoch_end):
                self._p(('Saving at epoch %d'%epoch))
                self._p(('savedir: %s' % savedir))
                try:
                    os.system('mkdir -p %s' % savedir)
                except:
                    pass
                if self.params['savemodel']:
                    self._saveModel(savedir)
                saveHDF5(os.path.join(savedir,'output.h5'), log)
                saveHDF5(os.path.join(savedir,'output_verbose.h5'), log_verbose)
                self.post_save_hook(**{k:v for k,v in locals().iteritems() if k!='self'})

        return {'output':log,'output_verbose':log_verbose}

    def _saveModel(self,savedir):
        """
        _saveModel: Save model to directory `savedir`.
        * model.tWeights saved to `savedir/tWeights.h5`
        * model.tOptWeights saved to `savedir/tOptWeights.h5`
        """
        tWeights = self.tWeights.apply(lambda x:x.get_value())
        tOptWeights = self.tOptWeights.apply(lambda x:x.get_value())

        tWeightsFile = os.path.join(savedir,'tWeights.h5')
        tOptWeightsFile = os.path.join(savedir,'tOptWeights.h5')

        saveHDF5(tWeightsFile,tWeights)
        saveHDF5(tOptWeightsFile,tOptWeights)

        self._p(('Saved model:'))
        self._p('tWeights: ' + tWeightsFile)
        self._p('tOptWeights: ' + tOptWeightsFile)

    def _loadModel(self,reloadDir,configFile):
        tWeightsFile = os.path.join(reloadDir,'tWeights.h5')
        tOptWeightsFile = os.path.join(reloadDir,'tOptWeights.h5')

        files = [paramsFile,tWeightsFile,tOptWeightsFile]
        for f in files: 
            assert os.path.exists(f),'cannot find %s' % f

        self.params = loadHDF5(configFile)
        self.tWeights = NestD(loadHDF5(tWeightsFile))
        self.tOptWeights = NestD(loadHDF5(tOptWeightsFile))


            

