import os
import runhpc

exptdir = '005_multiple_seeds_ApproxM2'
script = 'expt-ssl/train_ssl_vae_mnist.py'
rootdir = '/scratch/jmj418/theanomodels/experiments/mnist_ssl_vae'
session = exptdir
savedir = os.path.join(rootdir,exptdir)

#theano_flags='device=gpu{gpuid}'
theano_flags='device=gpu0,floatX=float32'
run_flags=['--savedir=%s'%savedir,
           '--epochs=500',
           '-sfreq 100',
           '-betaprior 0.2',
           '--batchnorm=True',
           '--separateBNrunningstats=True',
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
           '-model approxM2',
           #'-seed 1',
           '-rv 0.1']
var_flags = {
		'seed1':'-seed 1',
		'seed2':'-seed 2',
		'seed3':'-seed 3',
		'seed4':'-seed 4',
		'seed5':'-seed 5',
		'seed6':'-seed 6',
		'seed7':'-seed 7',
		'seed8':'-seed 8',
		'seed9':'-seed 9',
		'seed10':'-seed 10',
        }

memory = 10
walltime = 36
for tag,val in var_flags.iteritems():
    session_name = '%s_%s' % (session,tag)
    cmd='THEANO_FLAGS="{theano_flags},exception_verbosity=high" python -u {script} {run_flags}'.format(
            theano_flags=theano_flags+',compiledir=/home/jmj418/.theano/%s'%session_name,
            script=script,
            run_flags=' '.join(run_flags+[val]))
    runhpc.launch(rootdir,exptdir,session_name,cmd,memory,walltime)
    

