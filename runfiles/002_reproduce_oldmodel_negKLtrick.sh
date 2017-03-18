export ROOTDIR=output
export EXPTDIR=002_reproduce_oldmodel_negKLtrick
export SAVEDIR=$ROOTDIR/$EXPTDIR

THEANO_FLAGS=device=gpu1 python train.py --savedir=$SAVEDIR --epochs=2000 -betaprior 0.2 -rv 0.1 -sfreq 50 --batchnorm=True --seperateBNrunningstats=True -pl 3 -ph 200 -ql 3 -qh 300 -cw 128 -lr 5e-4 -ds 50 -al 1 -nl maxout2 --yKL=True --negKL=True -seed 1
