import os

exptdir = '005_GMM2'
script = 'train.py'
rootdir = '/scratch/jmj/vae_ssl/experiments/mnist_ssl_vae'
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
           '-nl maxout',
           '-pl 4',
           '-ph 200',
           '-ql 4',
           '-qh 300',
           '-al 2',
           #'-cw 128',
           '-lr 5e-4',
           '-ds 50',
           '-model GM2',
           '--track_params=True',
           '-seed 1',
           '-rv 0.1']
vary_flags = {
    'cw256_da50':'-cw 256 -da 50',
    'cw128_da50':'-cw 128 -da 50',
    'cw128_da200':'-cw 128 -da 200',
    #'cw128_da50_y1':'-cw 128 -da 50 -yl 1',
    #'cw128_da50_y2':'-cw 128 -da 50 -yl 2',
}

numgpus = 3
gpuctr = 0
for tag,val in vary_flags.iteritems():
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
    

