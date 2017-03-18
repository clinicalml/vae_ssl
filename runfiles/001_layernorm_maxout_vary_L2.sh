export ROOTDIR=output
export EXPTDIR=001_layernorm_maxout_vary_L2
export SAVEDIR=$ROOTDIR/$EXPTDIR

THEANO_FLAGS=device=gpu1 python train.py --savedir=$SAVEDIR --epochs=200 -sfreq 50 --layernorm=True -nl maxout -pl 3 -ph 200 -ql 3 -qh 300 -cw 128 -lr 5e-4 -seed 1 -rv 1e-3
THEANO_FLAGS=device=gpu1 python train.py --savedir=$SAVEDIR --epochs=200 -sfreq 50 --layernorm=True -nl maxout -pl 3 -ph 200 -ql 3 -qh 300 -cw 128 -lr 5e-4 -seed 1 -rv 1e-2
THEANO_FLAGS=device=gpu1 python train.py --savedir=$SAVEDIR --epochs=200 -sfreq 50 --layernorm=True -nl maxout -pl 3 -ph 200 -ql 3 -qh 300 -cw 128 -lr 5e-4 -seed 1 -rv 1e0
THEANO_FLAGS=device=gpu1 python train.py --savedir=$SAVEDIR --epochs=200 -sfreq 50 --layernorm=True -nl maxout -pl 3 -ph 200 -ql 3 -qh 300 -cw 128 -lr 5e-4 -seed 1 -rv 1e1
THEANO_FLAGS=device=gpu1 python train.py --savedir=$SAVEDIR --epochs=200 -sfreq 50 --layernorm=True -nl maxout -pl 3 -ph 200 -ql 3 -qh 300 -cw 128 -lr 5e-4 -seed 1 -rv 1e-2





