import os

exptdir = '001_LogGamma'
script = 'train.py'
rootdir = 'output'
session = exptdir
savedir = os.path.join(rootdir,exptdir)

theano_flags='device=gpu{gpuid}'
run_flags=[#'--savedir=%s'%savedir,
           '--epochs=300',
           '-sfreq 150',
           '-betaprior 0.2',
           '--batchnorm=True',
           #'--negKL=True',
           '--annealKL_Z=1',
           '--annealKL_alpha=1',
           '--annealCW=50000',
           '-nl maxout',
           '-pl 4',
           '-ph 200',
           '-ql 4',
           '-qh 300',
           '-al 2',
           #'-cw 128',
           '-lr 5e-4',
           '-ds 50',
           '-model LogGamma',
           #'--learn_posterior=True',
           #'-seed 1',
           '-rv 0.1']
vary_flags = {
    'cw128_seed1':'-cw 128 -seed 1'
}

numgpus = 3
gpuctr = 2
for tag,val in vary_flags.iteritems():
    gpu = gpuctr % numgpus
    gpuctr += 1
    session_name = '%s_%s' % (session,tag)
    cmd='THEANO_FLAGS="{theano_flags}" python {script} {run_flags}'.format(
            theano_flags=theano_flags.format(gpuid=gpu),
            script=script,
            run_flags=' '.join(run_flags+[val,'--savedir=%s'%os.path.join(savedir,tag)]))
    execute = 'tmux new -d -s {session_name}; tmux send-keys -t {session_name} "{cmd}" Enter'.format(**locals())
    #execute = 'tmux kill-session -t {session_name}'.format(**locals())
    print execute
    os.system(execute)
    

