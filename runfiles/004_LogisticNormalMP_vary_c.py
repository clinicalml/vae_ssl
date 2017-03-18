import os

exptdir = '004_LogisticNormalMP_vary_c'
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
           #'--negKL=True',
           '--annealKL=1',
           '-nl maxout',
           '-pl 4',
           '-ph 200',
           '-ql 4',
           '-qh 300',
           '-al 2',
           '-cw 128',
           '-lr 5e-4',
           '-ds 50',
           '-model LogisticNormalMP',
           #'--track_params=True',
           '-seed 1',
           '-rv 0.1']
vary_flag = '-lnmp {val}'
vary_vals = [0.5,1.0,1.5]

numgpus = 3
gpuctr = 0
for val in vary_vals:
    gpu = gpuctr % numgpus
    gpuctr += 1
    session_name = '%s_%s' % (session,val)
    session_name = session_name.replace('.','_')
    cmd='THEANO_FLAGS="{theano_flags},exception_verbosity=high" python {script} {run_flags}'.format(
            theano_flags=theano_flags.format(gpuid=gpu),
            script=script,
            run_flags=' '.join(run_flags+[vary_flag.format(val=val)]))
    execute = 'tmux new -d -s {session_name}; tmux send-keys -t {session_name} "{cmd}" Enter'.format(**locals())
    #execute = 'tmux kill-session -t {session_name}'.format(**locals())
    print execute
    os.system(execute)
    

