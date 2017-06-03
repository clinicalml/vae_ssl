import sys
from utils import Logger
import os
import numpy as np
from parse_args import params 
from utils import get_new_dirpath_as_timestring
from utils import track_and_print_time
import time

savedir = params['savedir']
if params['unique_id'] is not None:
    savedir += params['unique_id']
os.system('mkdir -p %s' % savedir)

# this will print stdout to both the terminal and the log file
# if mode='w', any existing logfile will be overwritten
logger = Logger(os.path.join(savedir,'logfile.log'),mode='w')

# send stderr to sys.stderr and logfile
with logger.send_stderr_to_logger():

    # send stdout to sys.stdout and logfile
    # note that the ProgressBar in AbstractModel will temporarily
    # stop sending print outputs to the logfile, because of the
    # carriage returns, which we don't want or need in our logfile,
    # because it would make the logfile look funny and hard to read
    with logger.send_stdout_to_logger():
        print '\nsavedir: %s\n' % savedir

        # import datasets here, because it calls theanomodels, which imports theano
        # and that will output gpu_usage stats, which we want to capture in the log
        from datasets import SemiSupervisedMNIST

        params['timestamp'] = time.time()

        print '\nparams:'
        for k in sorted(params):
            print '%s: %s' % (k,params[k])

        print '\nsetting numpy random seed to %s' % (params['seed']*10)
        np.random.seed(params['seed']*10)

        print ''
        with track_and_print_time(start_str='Loading dataset:'):
            dataset = SemiSupervisedMNIST(labeled_per_class=params['nlabelsperclass'])
            params['nclasses'] = dataset.nclasses
            params['dim_observations'] = dataset.dim_observations
            print dataset


        print ''
        with track_and_print_time('importing model'):
            if params['model'] == 'LogGamma':
                from models.LogGamma import LogGammaSemiVAE
                SSL_VAE_CLASS = LogGammaSemiVAE
            elif params['model'] == 'LogisticNormal':
                from models.LogisticNormal import LogisticNormalSemiVAE 
                SSL_VAE_CLASS = LogisticNormalSemiVAE 
            elif params['model'] == 'ApproxM2':
                from models.ApproxM2 import ApproxM2SemiVAE
                SSL_VAE_CLASS = ApproxM2SemiVAE
            elif params['model'] == 'ExactM2':
                from models.ExactM2 import ExactM2SemiVAE
                SSL_VAE_CLASS = ExactM2SemiVAE
            elif params['model'] in ['GumbelSoftmaxM2','STGumbelSoftmaxM2']:
                from models.GumbelSoftmaxM2 import GumbelSoftmaxM2SemiVAE 
                SSL_VAE_CLASS = GumbelSoftmaxM2SemiVAE
            elif params['model'] in ['GumbelSoftmax','STGumbelSoftmax']:
                from models.GumbelSoftmax import GumbelSoftmax 
                SSL_VAE_CLASS = GumbelSoftmax
            elif params['model'] in ['LogisticNormalM2','STLogisticNormalM2']:
                from models.LogisticNormalM2 import LogisticNormalM2SemiVAE 
                SSL_VAE_CLASS = LogisticNormalM2SemiVAE
            else:
                raise NameError('unhandled model type: %s' % params['model'])

        print ''
        with track_and_print_time('Building %s' % params['model']):
            vae = SSL_VAE_CLASS(params, 
                                configFile=os.path.join(savedir,params['configFile']), 
                                reloadDir=params['reloadDir'],
                                logfile=logger.log) 

        print ''
        with track_and_print_time('Running %s' % params['model']):
            output = vae.learn(  
                                dataset,
                                epoch_start=0 , 
                                epoch_end = params['epochs'], 
                                batchsize = params['batchsize'],
                                savedir = savedir,
                                savefreq = params['savefreq'],
                                evalfreq = params['evalfreq'],
                                maxiters = params['maxiters'],
                                collect_garbage = False,
                             )
