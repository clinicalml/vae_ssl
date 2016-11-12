import os

exptdir = '004_approxM2_get_samples'
script = 'train.py'
rootdir = '/scratch/jmj/vae_ssl/experiments/mnist_ssl_vae'
session = exptdir
savedir = os.path.join(rootdir,exptdir)

theano_flags='device=gpu{gpuid}'
run_flags=['--savedir=%s'%savedir,
           '--epochs=300',
           '-sfreq 300',
           '-betaprior 0.2',
           #'--bilinear=True',
           #'-nl maxout',
           #'-pl 4',
           '-ph 200',
           #'-ql 4',
           '-qh 300',
           #'-al 2',
           '-ankl 1',  #annealing
           '-cw 128',
           '-lr 5e-4',
           '-ds 50',
           '-seed 1',
           '-rv 0.1',
           '-model approxM2',
           '--track_params=False',
           ]
vary_flag = '{val}'
vary_vals = {
'BN_pl4ql4':'-nl maxout -pl 4 -ql 4 -al 2 --batchnorm=True --separateBNrunningstats=True',
}
start_gpu = 2
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
    

