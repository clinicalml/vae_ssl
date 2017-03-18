import os

exptdir = '004_LogisticNormal_vary_bxy'
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
           '-model LogisticNormal',
           #'--track_params=True',
           '-seed 1',
           '-rv 0.1']
vary_flags = {
    #'bxy2':'-bxy 2',
    #'bxy5':'-bxy 5',
    #'bxy50':'-bxy 50',
    #'bxy1000':'-bxy 1000',
    'bxy2_cw256':'-bxy 2 -cw 256',
    'bxy2_cw512':'-bxy 2 -cw 512',
    'bxy4_cw256':'-bxy 4 -cw 256',
    'bxy4_cw512':'-bxy 4 -cw 512',
    'bxy50_cw256':'-bxy 50 -cw 256',
    'bxy50_cw512':'-bxy 50 -cw 512',
}

numgpus = 3
gpuctr = 2
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
    

