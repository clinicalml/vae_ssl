import os,time
import numpy as np
from theanomodels.datasets.load import loadDataset
from theanomodels.utils.misc import removeIfExists,createIfAbsent,displayTime
from parse_args import params 
from datasets import SemiSupervisedMNIST

start_time = time.time()
os.system('mkdir -p %s' % params['savedir'])
dataset = SemiSupervisedMNIST(labeled_per_class=params['nlabelsperclass'])
params['nclasses'] = dataset.nclasses
params['dim_observations'] = dataset.dim_observations

for k in sorted(params):
	print '%s: %s' % (k,params[k])

#Setup VAE Model (or reload from existing savefile)

if params['model'] == 'LogGamma':
    from models.LogGamma import LogGammaSemiVAE
    SSL_VAE_CLASS = LogGammaSemiVAE
elif params['model'] == 'LogGammaReverse':
    from models.reverse import LogGammaSemiVAEReverse
    SSL_VAE_CLASS = LogGammaSemiVAEReverse
elif params['model'] == 'DirichletSeparate':
    from models.separate import LogGammaSemiVAESeparate
    SSL_VAE_CLASS = LogGammaSemiVAESeparate
elif params['model'] == 'approxM2':
    from models.approxM2 import SemiVAE
    SSL_VAE_CLASS = SemiVAE
elif params['model'] == 'exactM2':
    from models.exactM2 import ExactSemiVAE
    SSL_VAE_CLASS = ExactSemiVAE
elif params['model'] == 'exactM2_debug':
    from models.vae_ssl_exactM2 import ExactSemiVAE
    SSL_VAE_CLASS = ExactSemiVAE
elif params['model'] in ['GumbelSoftmaxM2','STGumbelSoftmaxM2']:
    from models.GumbelSoftmaxM2 import SemiVAE
    SSL_VAE_CLASS = SemiVAE
elif params['model'] in ['GumbelSoftmax','STGumbelSoftmax']:
    from models.GumbelSoftmax import GumbelSoftmax 
    SSL_VAE_CLASS = GumbelSoftmax
elif params['model'] in ['LogisticNormalM2','STLogisticNormalM2']:
    from models.LogisticNormalM2 import LogisticNormalM2 
    SSL_VAE_CLASS = LogisticNormalM2 
elif params['model'] in ['LogisticNormal_warped','STLogisticNormal_warped']:
    from models.LogisticNormal_warped import LogisticNormal 
    SSL_VAE_CLASS = LogisticNormal
elif params['model'] == 'MixDirPrior':
    from models.MixDirPrior import MixDirPriorSemiVAE
    SSL_VAE_CLASS = MixDirPriorSemiVAE 
elif params['model'] == 'DirichletMixture':
    from models.DirichletMixture import DirichletMixtureSemiVAE 
    SSL_VAE_CLASS = DirichletMixtureSemiVAE 
elif params['model'] == 'LogGammaMixture':
    from models.LogGammaMixture import LogGammaMixtureSemiVAE 
    SSL_VAE_CLASS = LogGammaMixtureSemiVAE 
elif params['model'] == 'LogGammaLatentMixture':
    from models.LogGammaLatentMixture import LogGammaLatentMixtureSemiVAE 
    SSL_VAE_CLASS = LogGammaLatentMixtureSemiVAE 
elif params['model'] == 'GM':
    from models.GaussianMixture import GaussianMixtureSemiVAE 
    SSL_VAE_CLASS = GaussianMixtureSemiVAE 
elif params['model'] == 'GM2':
    from models.GaussianMixture2 import GaussianMixtureSemiVAE 
    SSL_VAE_CLASS = GaussianMixtureSemiVAE 
elif params['model'] == 'Dirichlet2':
    from models.Dirichlet import DirichletSemiVAE 
    SSL_VAE_CLASS = DirichletSemiVAE 
elif params['model'] == 'LogisticNormal':
    from models.LogisticNormal import LogisticNormalSemiVAE 
    SSL_VAE_CLASS = LogisticNormalSemiVAE 
elif params['model'] == 'LogisticNormal_fp':
	from models.LogisticNormal_fp import LogisticNormalSemiVAE 
    SSL_VAE_CLASS = LogisticNormalSemiVAE 
elif params['model'] == 'LNprd':
    from models.LogisticNormal_prod import LogisticNormalSemiVAE 
    SSL_VAE_CLASS = LogisticNormalSemiVAE 
elif params['model'] == 'LogisticNormalMP':
    from models.LogisticNormalMP import LogisticNormalMPSemiVAE 
    SSL_VAE_CLASS = LogisticNormalMPSemiVAE 
else:
    raise NameError('unhandled model type: %s' % params['model'])
displayTime('import vae_ssl',start_time, time.time())

start_time = time.time()
reloadDir = params['reloadDir']
configFile = os.path.join(params['savedir'],params['configFile'])
vae = SSL_VAE_CLASS(params, configFile, reloadDir) 
displayTime('Building vae_ssl',start_time, time.time())

start_time = time.time()
output = vae.learn(  
					dataset,
					epoch_start=0 , 
					epoch_end = params['epochs'], 
					batch_size = params['batch_size'],
					savedir = params['savedir'],
					savefreq = params['savefreq'],
					evalfreq = params['evalfreq'],
					max_iters = params['maxiters'],
					collect_garbage = None,
				 )
displayTime('Running VAE_SSL',start_time, time.time())
