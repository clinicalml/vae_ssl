export ROOTDIR=output
export EXPTDIR=000_layernorm_maxout
export SAVEDIR=$ROOTDIR/$EXPTDIR

THEANO_FLAGS=device=gpu2 python train.py --savedir=$SAVEDIR --epochs=500 -sfreq 50 --layernorm=True -nl maxout -pl 3 -ph 200 -ql 3 -qh 300 -cw 128 -lr 5e-4 -seed 1
