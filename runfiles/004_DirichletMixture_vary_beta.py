import os

exptdir = '004_DirichletMixture_vary_beta'
script = 'train.py'
rootdir = '/scratch/jmj/vae_ssl/experiments/mnist_ssl_vae'
session = exptdir
savedir = os.path.join(rootdir,exptdir)

theano_flags='device=gpu{gpuid}'
run_flags=['--savedir=%s'%savedir,
           '--epochs=300',
           '-sfreq 50',
           #'-betaprior 0.2',
           '--batchnorm=True',
           '--separateBNrunningstats=True',
           '--annealKL=1',
           #'--negKL=True',
           '-nl maxout',
           '-pl 4',
           '-ql 4',
           '-al 2',
           '-ph 200',
           '-qh 300',
           '-cw 128',
           '-lr 5e-4',
           '-ds 50',
           '-model DirichletMixture',
           '-seed 1',
           '-rv 0.1',
           ]
var_flags = {
        'beta1e-3':'-betaprior 1e-3 -finalbeta 1e-3',
        'beta5e-3':'-betaprior 5e-3 -finalbeta 5e-3',
        'beta1e-2':'-betaprior 1e-2 -finalbeta 1e-2',
        'beta5e-2':'-betaprior 5e-2 -finalbeta 5e-2',
        'beta1e-1':'-betaprior 1e-1 -finalbeta 1e-1',
        'beta5e-1':'-betaprior 5e-1 -finalbeta 5e-1',
        }

numgpus = 3
gpuctr = 0
for tag,val in var_flags.iteritems():
    gpu = gpuctr % numgpus
    gpuctr += 1
    session_name = '%s_%s' % (session,tag)
    cmd='THEANO_FLAGS="{theano_flags},exception_verbosity=high" python {script} {run_flags}'.format(
            theano_flags=theano_flags.format(gpuid=gpu),
            script=script,
            run_flags=' '.join(run_flags+[val]))
    execute = 'tmux new -d -s {session_name}; tmux send-keys -t {session_name} "{cmd}" Enter'.format(**locals())
    #execute = 'tmux kill-session -t {session_name}'.format(**locals())
    print execute
    os.system(execute)
    
