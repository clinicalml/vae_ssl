import os

session_name = '001_layernorm_maxout_vary_LR'
script = 'train.py'
rootdir = '/scratch/jmj/vae_ssl/experiments/mnist_ssl_vae'
exptdir = '001_layernorm_maxout_vary_LR'
savedir = os.path.join(rootdir,exptdir)

theano_flags=['device=gpu0']
run_flags=['--savedir=%s'%savedir,
           '--epochs=500',
           '-sfreq 50',
           '--layernorm=True',
           '-nl maxout',
           '-pl 3',
           '-ph 200',
           '-ql 3',
           '-qh 300',
           '-cw 128',
           '-seed 1',
           '-rv 0.1']
vary_flag = '-lr {val}'
vary_vals = [5e-3,1e-3,5e-4,1e-4,5e-5]

for val in vary_vals:
    session_name = '%s_%0.0e' % (exptdir,val)
    cmd='THEANO_FLAGS={theano_flags} python {script} {run_flags}'.format(
            theano_flags=''.join(theano_flags),
            script=script,
            run_flags=' '.join(run_flags+[vary_flag.format(val=val)]))
    execute = 'tmux new -d -s {session_name}; tmux send-keys -t {session_name} "{cmd}" Enter'.format(**locals())
    #execute = 'tmux kill-session -t {session_name}'.format(**locals())
    print execute
    os.system(execute)
    

