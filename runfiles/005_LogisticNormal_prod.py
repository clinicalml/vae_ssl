import os

exptdir = '005_LNprd'
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
           #'--annealKL_Z=1',
           #'--annealKL_alpha=1',
           '--annealCW=1',
           '-nl maxout',
           '-pl 4',
           '-ph 200',
           '-ql 4',
           '-qh 300',
           '-al 2',
           #'-cw 128',
           '-lr 5e-4',
           '-ds 50',
           '-model LNprd',
           #'--learn_posterior=True',
           #'--track_params=True',
           '-seed 1',
           '-rv 0.1']
vary_flags = {
    #'cw256_aKL100':'-cw 256 --annealKL_Z=50000 --annealKL_alpha=50000',
    #'cw256_aKL100_hy2':'-cw 256 --annealKL_Z=50000 --annealKL_alpha=50000 --y_inference_layers=2 --learn_posterior=True',
    'cw256_aKL100_hy2_v2':'-cw 256 --annealKL_Z=50000 --annealKL_alpha=50000 --y_inference_layers=2',
}

numgpus = 3
gpuctr = 1
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
    

