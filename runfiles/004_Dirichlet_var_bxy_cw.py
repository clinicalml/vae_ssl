import os

exptdir = '005_Dirichlet'
script = 'train.py'
rootdir = '/scratch/jmj/vae_ssl/experiments/mnist_ssl_vae'
session = exptdir
savedir = os.path.join(rootdir,exptdir)

theano_flags='device=gpu{gpuid}'
run_flags=['--savedir=%s'%savedir,
           '--epochs=300',
           '-sfreq 50',
           '-betaprior 0.2',
           '--batchnorm=True',
           '--separateBNrunningstats=True',
           '--annealKL=1',
           '--negKL=True',
           '-nl maxout',
           '-pl 4',
           '-ql 4',
           '-al 2',
           '-ph 200',
           '-qh 300',
           '-cw 128',
           '-lr 5e-4',
           '-ds 50',
           #'-model Dirichlet',
           '-seed 1',
           '-rv 0.1',
           ]
var_flags = {
        'cw256':'-cw 256 -annealBP 50000 -finalbeta 5e-3 --negKL=True',
        'cw512':'-cw 512 -annealBP 50000 -finalbeta 5e-3 --negKL=True',
        'cw8192':'-cw 8192 -annealBP 50000 -finalbeta 5e-3 --negKL=True',
        'bxy2':'-bxy 2 -annealBP 50000 -finalbeta 5e-3 --negKL=True',
        'bxy4':'-bxy 4 -annealBP 50000 -finalbeta 5e-3 --negKL=True',
        'bxy2_cw256':'-bxy 2 -cw 256 -annealBP 50000 -finalbeta 5e-3 --negKL=True',
        'bxy2_cw512':'-bxy 2 -cw 512 -annealBP 50000 -finalbeta 5e-3 --negKL=True',
        }

numgpus = 3
gpuctr = 1
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
    

