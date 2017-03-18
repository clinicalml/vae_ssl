import os

exptdir = '003_collectstats_bilinear2_anneal_KL_loggamma'
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
'hzl1_BN_KL1e-3':'-hzl 1 -zl 2 -pzl 1 --batchnorm=True --separateBNrunningstats=True -kllg 1e-3',
'hzl1_BN_KL1e-2':'-hzl 1 -zl 2 -pzl 1 --batchnorm=True --separateBNrunningstats=True -kllg 1e-2',
'hzl1_BN_KL1e-1':'-hzl 1 -zl 2 -pzl 1 --batchnorm=True --separateBNrunningstats=True -kllg 1e-1',
'hzl1_BN_KL1e-0':'-hzl 1 -zl 2 -pzl 1 --batchnorm=True --separateBNrunningstats=True',

'hzl1_LN_KL1e-3':'-hzl 1 -zl 2 -pzl 1 --layernorm=True -kllg 1e-3',
'hzl1_LN_KL1e-2':'-hzl 1 -zl 2 -pzl 1 --layernorm=True -kllg 1e-2',
'hzl1_LN_KL1e-1':'-hzl 1 -zl 2 -pzl 1 --layernorm=True -kllg 1e-1',
'hzl1_LN_KL1e-0':'-hzl 1 -zl 2 -pzl 1 --layernorm=True',

'pl4ql4_BN_KL1e-3':'--batchnorm=True --separateBNrunningstats=True -kllg 1e-3',
'pl4ql4_BN_KL1e-2':'--batchnorm=True --separateBNrunningstats=True -kllg 1e-3',
'pl4ql4_BN_KL1e-1':'--batchnorm=True --separateBNrunningstats=True -kllg 1e-3',
'pl4ql4_BN_KL1e-0':'--batchnorm=True --separateBNrunningstats=True',

'pl4ql4_LN_KL1e-3':'--layernorm=True -kllg 1e-3',
'pl4ql4_LN_KL1e-2':'--layernorm=True -kllg 1e-3',
'pl4ql4_LN_KL1e-1':'--layernorm=True -kllg 1e-3',
'pl4ql4_LN_KL1e-0':'--layernorm=True',
}
start_gpu = 0
ignore = []

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
    

