import os

exptdir = '003_collectstats_logpxy_discrete'
script = 'train.py'
rootdir = '/scratch/jmj/vae_ssl/experiments/mnist_ssl_vae'
session = exptdir
savedir = os.path.join(rootdir,exptdir)

theano_flags='device=gpu{gpuid}'
run_flags=['--savedir=%s'%savedir,
           '--epochs=300',
           '-sfreq 100',
           '-betaprior 0.2',
           #'--bilinear=True',
           '-nl maxout',
           '-pl 4',
           '-ph 200',
           '-ql 4',
           '-qh 300',
           '-al 2',
           '-ankl 1',  #annealing
           '-cw 128',
           '-lr 5e-4',
           '-ds 50',
           '-seed 1',
           '-rv 0.1'
           ]
vary_flag = '{val}'
vary_vals = {
'pl4ql4_nKL_BN_sh3':'--logpxy_discrete=True --negKL=True --batchnorm=True --separateBNrunningstats=True --sharpening=3',
'pl4ql4_BN_sh3':'--logpxy_discrete=True --batchnorm=True --separateBNrunningstats=True --sharpening=3',
'pl4ql4_BN_sh3_KL1e-2':'--logpxy_discrete=True --batchnorm=True --separateBNrunningstats=True --sharpening=3 -kllg 1e-2',
#'pl4ql4_nKL_BN':'--logpxy_discrete=True --negKL=True --batchnorm=True --separateBNrunningstats=True',
#'pl4ql4_nKL_LN':'--logpxy_discrete=True --negKL=True --layernorm=True',
#'pl4ql4_BN':'--logpxy_discrete=True --batchnorm=True --separateBNrunningstats=True',
#'pl4ql4_LN':'--logpxy_discrete=True --layernorm=True',
#'pl4ql4_nKL_BN_bilinear':'--logpxy_discrete=True --negKL=True --batchnorm=True --separateBNrunningstats=True --bilinear=True',
#'pl4ql4_nKL_LN_bilinear':'--logpxy_discrete=True --negKL=True --layernorm=True --bilinear=True',
}
start_gpu = 1
ignore = [2]

def getGPU(gpuctr,numgpus,ignore=[]):
    assert sorted(ignore) != range(numgpus), 'cannot ignore all gpus!'
    gpu = gpuctr % numgpus
    gpuctr += 1
    while gpu in ignore:
        gpu, gpuctr = getGPU(gpuctr,numgpus)
    return gpu,gpuctr
numgpus = 3
gpuctr = start_gpu
for tag,val in vary_vals.iteritems():
    gpu,gpuctr = getGPU(gpuctr,numgpus,ignore)
    session_name = '%s_%s' % (session,tag)
    cmd='THEANO_FLAGS={theano_flags} python {script} {run_flags}'.format(
            theano_flags=theano_flags.format(gpuid=gpu),
            script=script,
            run_flags=' '.join(run_flags+[vary_flag.format(val=val)]))
    execute = 'tmux new -d -s {session_name}; tmux send-keys -t {session_name} "{cmd}" Enter'.format(**locals())
    #execute = 'tmux kill-session -t {session_name}'.format(**locals())
    print execute
    os.system(execute)
    

