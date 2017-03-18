export ROOTDIR=output
export EXPTDIR=debug
export SAVEDIR=$ROOTDIR/$EXPTDIR

#THEANO_FLAGS=device=gpu2 python train.py --savedir=$SAVEDIR --epochs=200 -sfreq 2 -efreq 1 -rv 0 --batchnorm=True --separateBNrunningstats=True -seed 1 -ql 2 -qh 3 -pl 2 -ph 3 -ds 2 
#THEANO_FLAGS=device=gpu2 python train.py --savedir=$SAVEDIR --epochs=5 -sfreq 2 -efreq 1 -rv 0 --batchnorm=True --separateBNrunningstats=True -seed 1 -ql 2 -qh 3 -pl 2 -ph 3 -ds 2 -zl 2 -hzl 2 -pzl 2 --track_params=True --model=approxM2  --negKL=True
#THEANO_FLAGS=device=gpu2 python train.py --savedir=$SAVEDIR --epochs=5 -sfreq 20 -efreq 10 -rv 0.1 --batchnorm=True --separateBNrunningstats=True --negKL=True -seed 1 -ql 3 -qh 300 -pl 3 -ph 200 -ds 50 -zl 2 -al 1  --track_params=True -nl maxout2 
#THEANO_FLAGS=device=gpu2 python train.py --savedir=$SAVEDIR --epochs=5 -sfreq 1 -efreq 1 -rv 0 --batchnorm=True --separateBNrunningstats=True -seed 1 -ql 2 -qh 3 -pl 2 -ph 3 -ds 2 -zl 2 -hzl 2 -pzl 2 --track_params=True -mbw 1 --negKL=True
THEANO_FLAGS=device=gpu2 python train.py --savedir=$SAVEDIR --epochs=10 -sfreq 1 -efreq 1 -rv 0 --batchnorm=True --separateBNrunningstats=True -seed 1 -ql 2 -qh 3 -pl 2 -ph 3 -ds 2 -zl 2 -hzl 2 -pzl 2 --track_params=False -annealBP 50000 -seed 1 -model GumbelSoftmaxM2 --savemodel=False

