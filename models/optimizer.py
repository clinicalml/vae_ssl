#LICENCE : for adam optimizer (Modified from https://gist.github.com/Newmu/acb738767acb4788bac3)
"""
The MIT License (MIT)
Copyright (c) 2015 Alec Radford
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""    


import theano
import numpy as np
from theano import config
from collections import OrderedDict
import theano.tensor as T
"""
                                            OPTIMIZERS FOR THEANO
"""
  

    
"""
                                            UTILITY FUNCTIONS
"""
    

def regularize(cost, params, reg_val, reg_type):
    """
    Return a theano cost
    cost: cost to regularize
    params: list of parameters
    reg_val: multiplier for regularizer
    reg_type: accepted types of regularizer 'l1','l2'
    pnorm_str: simple regex to exclude parameters not satisfying regex
    """
    l1 = lambda p: T.sum(abs(p))
    l2 = lambda p: T.sum(p**2)
    rFxn = {}
    rFxn['l1']=l1
    rFxn['l2']=l2
    
    if reg_type=='l2' or reg_type=='l1':
        assert reg_val is not None,'Expecting reg_val to be specified'
        print "<< Reg:("+reg_type+') Reg. Val:('+str(reg_val)+') >>'
        regularizer= theano.shared(np.asarray(0).astype(config.floatX),name = 'reg_norm', borrow=True)
        for p in params:
			regularizer += rFxn[reg_type](p)
			print ('<<<<<< Adding '+reg_type+' regularization for '+p.name)+' >>>>>>'
        return cost + reg_val*regularizer
    else:
        return cost

def normalize(grads, grad_norm):
    """
    grads: list of gradients
    grad_norm : None (or positive value)
    returns: gradients rescaled to satisfy norm constraints
    """
    #Check if we're clipping gradients
    if grad_norm is not None:
        assert grad_norm > 0, 'Must specify a positive value to normalize to'
        print '<<<<<< Normalizing Gradients to have norm (',grad_norm,') >>>>>>'
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(T.switch(g2 > (grad_norm**2), g/T.sqrt(g2)*grad_norm, g))
        return new_grads
    else:
        return grads

def rescale(grads, divide_grad):
    """
    grads : list of gradients
    divide_grad : scalar or theano variable
    returns: gradients divided by provided variable
    """
    if divide_grad is not None:
        print '<<<<<< Rescaling Gradients >>>>>>'
        new_grads = []
        for g in grads:
            new_grads.append(g/divide_grad)
        return new_grads
    else:
        return grads

    
    
    
"""
                                            OPTIMIZERS
"""
def adam(params, grads, lr=0.001, b1=0.1, b2=0.001, e=1e-8, opt_params=None):
    """
    ADAM Optimizer
    cost (to be minimized)
    params (list of parameters to take gradients with respect to)
    .... parameters specific to the optimization ...
    opt_params (if available, used to intialize the variables
    
    """
    updates = []
    
    restartOpt = False
    if opt_params is None:
        restartOpt = True
        opt_params=OrderedDict()
    
    #Track the optimization variable
    if restartOpt:
        i = theano.shared(np.asarray(0).astype(config.floatX),name ='opt_i',borrow=True)
        opt_params['opt_i'] = i
    else:
        i = opt_params['opt_i']
    
    #No need to reload these theano variables
    g_norm = theano.shared(np.asarray(0).astype(config.floatX),name = 'g_norm',borrow=True)
    p_norm = theano.shared(np.asarray(0).astype(config.floatX),name = 'p_norm',borrow=True)
    opt_norm = theano.shared(np.asarray(0).astype(config.floatX),name = 'opt_norm',borrow=True)
    
    #Initialization for ADAM
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    
    for p, g in zip(params, grads):
        if restartOpt:
            m = theano.shared(np.array(p.get_value() * 0.).astype(config.floatX),name = 'opt_m_'+p.name,borrow=True)
            v = theano.shared(np.array(p.get_value() * 0.).astype(config.floatX),name = 'opt_v_'+p.name,borrow=True)
            opt_params['opt_m_'+p.name] = m
            opt_params['opt_v_'+p.name] = v
        else:
            m = opt_params['opt_m_'+p.name] 
            v = opt_params['opt_v_'+p.name]
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
        #Update norms
        g_norm += (g**2).sum()
        p_norm += (p**2).sum() 
        opt_norm+=(m**2).sum() + (v**2).sum()
    updates.append((i, i_t))
    
    return updates, [T.sqrt(p_norm), T.sqrt(g_norm), T.sqrt(opt_norm)], opt_params 

def rmsprop(params, grads, lr=0.002, rho=0.9, epsilon = 1e-8, opt_params = None):
    """
    RMSPROP Optimizer
    cost (to be minimized)
    params (list of parameters to take gradients with respect to)
    .... parameters specific to the optimization ...
    opt_params (if available, used to intialize the variables
    returns: update list of tuples, 
             list of norms [0]: parameters [1]: gradients [2]: opt. params [3]: regularizer
             opt_params: dictionary containing all the parameters for the optimization
    """
    updates = []

    restartOpt = False
    if opt_params is None:
        restartOpt = True
        opt_params=OrderedDict()
    
    #No need to reload these
    g_norm = theano.shared(np.asarray(0).astype(config.floatX),name = 'g_norm',borrow=True)
    p_norm = theano.shared(np.asarray(0).astype(config.floatX),name = 'p_norm',borrow=True)
    opt_norm = theano.shared(np.asarray(0).astype(config.floatX),name = 'opt_norm',borrow=True)
    
    for p, g in zip(params,grads):
        if grad_range is not None:
            print '<<<<<< RMSPROP: Truncating Gradients in Range +-(',grad_range,') >>>>>>'
            g = T.clip(g,-grad_range, grad_range)
        
        if restartOpt:
            f_prev   = theano.shared(p.get_value()*0.,name = 'opt_fprev_'+p.name)
            r_prev   = theano.shared(p.get_value()*0.,name = 'opt_rprev_'+p.name)
            opt_params['opt_rprev_'+p.name] = r_prev
            opt_params['opt_fprev_'+p.name] = f_prev
        else:
            r_prev   = opt_params['opt_rprev_'+p.name]
            f_prev   = opt_params['opt_fprev_'+p.name]
        f_cur    = rho*f_prev+(1-rho)*g  
        r_cur    = rho*r_prev+(1-rho)*g**2
        updates.append((r_prev,r_cur))
        updates.append((f_prev,f_cur))
        
        lr_t = lr/T.sqrt(r_cur+f_cur**2+epsilon)
        p_t = p-(lr_t*g)
        updates.append((p,p_t))
        
        #Update norms
        g_norm += (g**2).sum()
        p_norm += (p**2).sum() 
        opt_norm+=(r_prev**2).sum()
    return updates, [T.sqrt(p_norm), T.sqrt(g_norm), T.sqrt(opt_norm)], opt_params

