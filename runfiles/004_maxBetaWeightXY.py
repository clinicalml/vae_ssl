import os

exptdir = '004_maxBetaWeightXY'
script = 'train.py'
rootdir = '/scratch/jmj/vae_ssl/experiments/mnist_ssl_vae'
session = exptdir
savedir = os.path.join(rootdir,exptdir)

theano_flags='device=gpu{gpuid}'
run_flags=['--savedir=%s'%savedir,
           '--epochs=300',
           '-sfreq 100',
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
           '--track_params=True',
           '-seed 1',
           '-rv 0.1']

vary_flags = {
        'mbw5':'-mbw 5',
        'mbw20':'-mbw 20',
        'mbw40':'-mbw 40',
        'mbw60':'-mbw 60',
        'mbwXY2':'-mbw 2 -mbxy 2',
        'mbwXY5':'-mbw 5 -mbxy 5',
        'mbwXY10':'-mbw 10 -mbxy 10',
        'mbwXY20':'-mbw 20 -mbxy 20',
        'mbwXY40':'-mbw 40 -mbxy 40',

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
    

