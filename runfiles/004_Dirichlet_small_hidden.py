import os

exptdir = '004_Dirichlet_small_hidden'
script = 'train.py'
rootdir = '/scratch/jmj/vae_ssl/experiments/mnist_ssl_vae'
session = exptdir
savedir = os.path.join(rootdir,exptdir)

theano_flags='device=gpu{gpuid}'
run_flags=['--savedir=%s'%savedir,
           '--epochs=300',
           '-sfreq 50',
           '--evalfreq=1',
           '-betaprior 0.2',
           '--batchnorm=True',
           '--separateBNrunningstats=True',
           '--annealKL=50000',
           '-nl maxout',
           '-pl 2',
           #'-ph 3',
           '-ql 2',
           #'-qh 3',
           #'-al 2',
           '-zl 2',
           '-hzl 2',
           '-pzl 2',
           '-cw 128',
           '-lr 5e-4',
           '-ds 2',
           #'-model LogisticNormal',
           '-seed 1',
           #'-rv 0.1',
           ]
var_flags = {
        'h3_rv0':'-ph 3 -qh 3 -rv 0',
        'h10_rv0':'-ph 10 -qh 10 -rv 0',
        'h50_rv0':'-ph 50 -qh 50 -rv 0',
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
    execute = 'tmux kill-session -t {session_name}'.format(**locals())
    print execute
    os.system(execute)
    

