import os,time
import numpy as np
from theanomodels.datasets.load import loadDataset
from theanomodels.utils.parse_args_vae_ssl import params 
from theanomodels.utils.misc import removeIfExists,createIfAbsent,mapPrint,saveHDF5,displayTime
import ipdb

dataset = 'mnist'
params['savedir']+='-'+dataset
createIfAbsent(params['savedir'])
dataset = loadDataset(dataset)

#Saving/loading
for k in ['dim_observations','data_type']:
    params[k] = dataset[k]
mapPrint('Options: ',params)

#Setup VAE Model (or reload from existing savefile)
start_time = time.time()
vae    = None

if params['model'] == 'Relaxed':
    from models.vae_ssl_LogGamma import LogGammaSemiVAE
    SSL_VAE_CLASS = LogGammaSemiVAE
elif params['model'] == 'RelaxedReverse':
    from models.vae_ssl_reverse import LogGammaSemiVAEReverse
    SSL_VAE_CLASS = LogGammaSemiVAEReverse
elif params['model'] == 'DirichletSeparate':
    from models.vae_ssl_separate import LogGammaSemiVAESeparate
    SSL_VAE_CLASS = LogGammaSemiVAESeparate
elif params['model'] == 'approxM2':
    from models.vae_ssl_approxM2 import SemiVAE
    SSL_VAE_CLASS = SemiVAE
elif params['model'] == 'exactM2':
    from models.vae_ssl_exactM2 import ExactSemiVAE
    SSL_VAE_CLASS = ExactSemiVAE
elif params['model'] == 'MixDirPrior':
    from models.vae_ssl_MixDirPrior import MixDirPriorSemiVAE
    SSL_VAE_CLASS = MixDirPriorSemiVAE 
elif params['model'] == 'DirichletMixture':
    from models.vae_ssl_DirichletMixture import DirichletMixtureSemiVAE 
    SSL_VAE_CLASS = DirichletMixtureSemiVAE 
elif params['model'] == 'LogGammaMixture':
    from models.vae_ssl_LogGammaMixture import LogGammaMixtureSemiVAE 
    SSL_VAE_CLASS = LogGammaMixtureSemiVAE 
elif params['model'] == 'LogGammaLatentMixture':
    from models.vae_ssl_LogGammaLatentMixture import LogGammaLatentMixtureSemiVAE 
    SSL_VAE_CLASS = LogGammaLatentMixtureSemiVAE 
elif params['model'] == 'GM':
    from models.vae_ssl_GaussianMixture import GaussianMixtureSemiVAE 
    SSL_VAE_CLASS = GaussianMixtureSemiVAE 
elif params['model'] == 'GM2':
    from models.vae_ssl_GaussianMixture2 import GaussianMixtureSemiVAE 
    SSL_VAE_CLASS = GaussianMixtureSemiVAE 
elif params['model'] == 'Dirichlet2':
    from models.vae_ssl_Dirichlet import DirichletSemiVAE 
    SSL_VAE_CLASS = DirichletSemiVAE 
elif params['model'] == 'LogisticNormal':
    from models.vae_ssl_LogisticNormal import LogisticNormalSemiVAE 
    SSL_VAE_CLASS = LogisticNormalSemiVAE 
elif params['model'] == 'LogisticNormal_fp':
    if params['modifiedBatchNorm']:
        from models.vae_ssl_LogisticNormal_fp_mbn import LogisticNormalSemiVAE 
    else:
        from models.vae_ssl_LogisticNormal_fp import LogisticNormalSemiVAE 
    SSL_VAE_CLASS = LogisticNormalSemiVAE 
elif params['model'] == 'LNprd':
    from models.vae_ssl_LogisticNormal_prod import LogisticNormalSemiVAE 
    SSL_VAE_CLASS = LogisticNormalSemiVAE 
elif params['model'] == 'LogisticNormalMP':
    from models.vae_ssl_LogisticNormalMP import LogisticNormalMPSemiVAE 
    SSL_VAE_CLASS = LogisticNormalMPSemiVAE 
else:
    raise NameError('unhandled model type: %s' % params['model'])
displayTime('import vae_ssl',start_time, time.time())

#Remove from params
start_time = time.time()
removeIfExists('./NOSUCHFILE')
reloadFile = params.pop('reloadFile')
savef      = os.path.join(params['savedir'],params['unique_id'],'seed-%s'%params['seed']) 
os.system('mkdir -p %s' % savef)
if os.path.exists(reloadFile):
    pfile=params.pop('paramFile')
    assert os.path.exists(pfile),pfile+' not found. Need paramfile'
    print 'Reloading trained model from : ',reloadFile
    print 'Assuming ',pfile,' corresponds to model'
    vae  = SSL_VAE_CLASS(params, paramFile = pfile, reloadFile = reloadFile) 
else:
    pfile= os.path.join(savef,'config.pkl')
    print 'Training model from scratch. Parameters in: ',pfile
    vae  = SSL_VAE_CLASS(params, paramFile = pfile)
displayTime('Building vae_ssl',start_time, time.time())

start_time = time.time()

replicate_K = 1

trainData = dataset['train'];validData = dataset['valid']

print 'setting numpy random seed to %s' % (params['seed']*10)
np.random.seed(params['seed']*10)
X = dataset['train']
Y = dataset['train_y'].astype('int32')
classes = range(params['nclasses'])
XL = []; YL = [];
for c in classes:
    sel = Y == c
    nc = sel.sum()
    Xc = X[sel]
    Yc = Y[sel]
    idx = np.arange(nc)
    np.random.shuffle(idx)
    Xc = Xc[idx[:params['nlabelsperclass']]]
    Yc = Yc[idx[:params['nlabelsperclass']]]
    XL.append(Xc)
    YL.append(Yc)
XL = np.vstack(XL)
YL = np.hstack(YL)
trainData = {'XU':X,'XL':XL,'YL':YL,'YU':Y}
validData = {'X':dataset['valid'],'Y':dataset['valid_y']}
testData = {'X':dataset['test'],'Y':dataset['test_y']}

savedata, samples = vae.learn(  trainData,
                                epoch_start=0 , 
                                epoch_end  = params['epochs'], 
                                batch_size = params['batch_size'],
                                savefreq   = params['savefreq'],
                                evalfreq   = params['evalfreq'],
                                savedir    = savef,
                                shuffle    = True,
                                dataset_eval= validData,
                                replicate_K= replicate_K
                                )
total_time = time.time()-start_time
displayTime('Running VAE_SSL',start_time, time.time())
t_outputs = vae.evaluateBound(testData, params['batch_size'], S=10)
savedata['test'] = t_outputs
savedata['time'] = total_time

#Save file log file
filename=os.path.join(savef,'final.h5')
saveHDF5(filename,savedata)
saveHDF5(os.path.join(savef,'samples.h5'),samples)
print 'saved to: %s' % filename
print 'Test Cost: ',savedata['test']['cost']
print 'Test Accuracy: ',savedata['test']['accuracy']
print 'Test Bound: ',savedata['test']['bound']

#import ipdb;
#ipdb.set_trace()
