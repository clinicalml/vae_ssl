import os

exptdir = '003_collectstats_bilinear2_newlayers'
script = 'train.py'
rootdir = '/scratch/jmj/vae_ssl/experiments/mnist_ssl_vae'
session = exptdir
savedir = os.path.join(rootdir,exptdir)

theano_flags='device=gpu{gpuid}'
run_flags=['--savedir=%s'%savedir,
           '--epochs=300',
           '-sfreq 100',
           '-betaprior 0.2',
           '--bilinear=True',
           '-nl maxout2',
           '-pl 3',
           '-ph 200',
           '-ql 3',
           '-qh 300',
           '-al 3',
           '-cw 128',
           '-lr 5e-4',
           '-ds 50',
           '-seed 1',
           '-rv 0.1'
           ]
vary_flag = '{val}'
vary_vals = {
        #'hzl1-BN':'-hzl 1 -zl 2 -pzl 1 --batchnorm=True --separateBNrunningstats=True',
        #'hzl2-BN':'-hzl 2 -zl 1 -pzl 1 --batchnorm=True --separateBNrunningstats=True',
        #'pzl2-BN':'-hzl 1 -zl 2 -pzl 2 --batchnorm=True --separateBNrunningstats=True',
        #'hzl1-mBN':'-hzl 1 -zl 2 -pzl 1 --batchnorm=True --separateBNrunningstats=True --modifiedBatchNorm=True',
        #'hzl1-LN':'-hzl 1 -zl 2 -pzl 1 --layernorm=True',
        #'hzl1':'-hzl 1 -zl 2 -pzl 1'
        'hzl1-BN-nKL':'-hzl 1 -zl 2 -pzl 1 --batchnorm=True --separateBNrunningstats=True --negKL=True',
        }

def getGPU(gpuctr,numgpus,ignore=[]):
    assert sorted(ignore) != range(numgpus), 'cannot ignore all gpus!'
    gpu = gpuctr % numgpus
    gpuctr += 1
    while gpu in ignore:
        gpu, gpuctr = getGPU(gpuctr,numgpus)
    return gpu,gpuctr
numgpus = 3
gpuctr = 2
ignore = []
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
    

