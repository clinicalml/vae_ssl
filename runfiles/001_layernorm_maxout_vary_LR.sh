export ROOTDIR=output
export EXPTDIR=001_layernorm_maxout_vary_LR
export SAVEDIR=$ROOTDIR/$EXPTDIR

THEANO_FLAGS=device=gpu0 python train.py --savedir=$SAVEDIR --epochs=200 -sfreq 50 --layernorm=True -nl maxout -pl 3 -ph 200 -ql 3 -qh 300 -cw 128 -rv 0.1 -seed 1 -lr 5e-1 
THEANO_FLAGS=device=gpu0 python train.py --savedir=$SAVEDIR --epochs=200 -sfreq 50 --layernorm=True -nl maxout -pl 3 -ph 200 -ql 3 -qh 300 -cw 128 -rv 0.1 -seed 1 -lr 5e-2
THEANO_FLAGS=device=gpu0 python train.py --savedir=$SAVEDIR --epochs=200 -sfreq 50 --layernorm=True -nl maxout -pl 3 -ph 200 -ql 3 -qh 300 -cw 128 -rv 0.1 -seed 1 -lr 5e-3
THEANO_FLAGS=device=gpu0 python train.py --savedir=$SAVEDIR --epochs=200 -sfreq 50 --layernorm=True -nl maxout -pl 3 -ph 200 -ql 3 -qh 300 -cw 128 -rv 0.1 -seed 1 -lr 5e-5





