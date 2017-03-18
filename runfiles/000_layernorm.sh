export ROOTDIR=output
export EXPTDIR=000_layernorm
export SAVEDIR=$ROOTDIR/$EXPTDIR

THEANO_FLAGS=device=gpu2 python train.py --savedir=$SAVEDIR --epochs=200 -sfreq 200 -rv 0 --layernorm=True -seed 1
#THEANO_FLAGS=device=gpu2 python train.py --savedir=$SAVEDIR --epochs=200 -sfreq 200 -rv 0 --layernorm=True -seed 2
#THEANO_FLAGS=device=gpu2 python train.py --savedir=$SAVEDIR --epochs=200 -sfreq 200 -rv 0 --layernorm=True -seed 3
#THEANO_FLAGS=device=gpu2 python train.py --savedir=$SAVEDIR --epochs=200 -sfreq 200 -rv 0 --layernorm=True -seed 4
#THEANO_FLAGS=device=gpu2 python train.py --savedir=$SAVEDIR --epochs=200 -sfreq 200 -rv 0 --layernorm=True -seed 5
#THEANO_FLAGS=device=gpu2 python train.py --savedir=$SAVEDIR --epochs=200 -sfreq 200 -rv 0 --layernorm=True -seed 6
#THEANO_FLAGS=device=gpu2 python train.py --savedir=$SAVEDIR --epochs=200 -sfreq 200 -rv 0 --layernorm=True -seed 7
#THEANO_FLAGS=device=gpu2 python train.py --savedir=$SAVEDIR --epochs=200 -sfreq 200 -rv 0 --layernorm=True -seed 8
#THEANO_FLAGS=device=gpu2 python train.py --savedir=$SAVEDIR --epochs=200 -sfreq 200 -rv 0 --layernorm=True -seed 9
#THEANO_FLAGS=device=gpu2 python train.py --savedir=$SAVEDIR --epochs=200 -sfreq 200 -rv 0 --layernorm=True -seed 10
