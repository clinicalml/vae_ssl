import os

exptdir = '006_GumbelSoftmax'
script = 'train.py'
rootdir = 'output'
session = exptdir
savedir = os.path.join(rootdir,exptdir)

theano_flags='device=gpu{gpuid}'
run_flags=['--savedir=%s'%savedir,
           '--epochs=300',
           '-sfreq 150',
           '-betaprior 0.2',
           '--batchnorm=True',
           '--separateBNrunningstats=True',
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
           '-model GumbelSoftmax',
           #'--learn_posterior=True',
           '--track_params=False',
           '-seed 1',
           '-rv 0.1']
vary_flags = {
    #'shp3e0_cw128':'--sharpening=3 -cw 128 --posterior_c=-6',
    #'shp1e0_cw128':'--sharpening=1 -cw 128 --posterior_c=-6',
    #'shp1e1_cw128':'--sharpening=10 -cw 128 --posterior_c=-6',
    'shp2e0_cw128':'--sharpening=2 -cw 128 --posterior_c=-6 --annealSharpening=50000',
}

numgpus = 3
gpuctr = 0
for tag,val in vary_flags.iteritems():
    gpu = gpuctr % numgpus
    gpuctr += 1
    session_name = '%s_%s' % (session,tag)
    cmd='THEANO_FLAGS="{theano_flags}" python {script} {run_flags}'.format(
            theano_flags=theano_flags.format(gpuid=gpu),
            script=script,
            run_flags=' '.join(run_flags+[val]))
    execute = 'tmux new -d -s {session_name}; tmux send-keys -t {session_name} "{cmd}" Enter'.format(**locals())
    #execute = 'tmux kill-session -t {session_name}'.format(**locals())
    print execute
    os.system(execute)
    

