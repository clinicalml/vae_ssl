export ROOTDIR=output
export EXPTDIR=001_layernorm_maxout_vary_cw
export SAVEDIR=$ROOTDIR/$EXPTDIR

#THEANO_FLAGS=device=gpu2 python train.py --savedir=$SAVEDIR --epochs=200 -sfreq 50 --layernorm=True -nl maxout -pl 3 -ph 200 -ql 3 -qh 300 -cw 1 -lr 5e-4 -seed 1
#THEANO_FLAGS=device=gpu2 python train.py --savedir=$SAVEDIR --epochs=200 -sfreq 50 --layernorm=True -nl maxout -pl 3 -ph 200 -ql 3 -qh 300 -cw 8 -lr 5e-4 -seed 1
#THEANO_FLAGS=device=gpu2 python train.py --savedir=$SAVEDIR --epochs=200 -sfreq 50 --layernorm=True -nl maxout -pl 3 -ph 200 -ql 3 -qh 300 -cw 64 -lr 5e-4 -seed 1
#THEANO_FLAGS=device=gpu2 python train.py --savedir=$SAVEDIR --epochs=200 -sfreq 50 --layernorm=True -nl maxout -pl 3 -ph 200 -ql 3 -qh 300 -cw 256 -lr 5e-4 -seed 1
THEANO_FLAGS=device=gpu2 python train.py --savedir=$SAVEDIR --epochs=200 -sfreq 50 --layernorm=True -nl maxout -pl 3 -ph 200 -ql 3 -qh 300 -cw 512 -lr 5e-4 -seed 1
#THEANO_FLAGS=device=gpu2 python train.py --savedir=$SAVEDIR --epochs=200 -sfreq 50 --layernorm=True -nl maxout -pl 3 -ph 200 -ql 3 -qh 300 -cw 128 -lr 5e-4 -seed 1





