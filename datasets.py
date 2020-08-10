import collections
from theanomodels.datasets.load import loadDataset
import numpy as np
from models import NestD

def infer_slice(idx,n):
    assert isinstance(idx,slice),'idx must be a slice'
    start = idx.start or idx.start if idx.start >= 0 else n+idx.start
    stop = idx.stop or idx.stop if idx > 0 else n+idx.stop
    step = idx.step or 1
    return range(start,stop,step)


class Data(object):

    def __init__(self,data={},n=None,*args,**kwargs):
        """
        Converts data to a NestD object.
        data: {dict,NestD}
        n: {int}
        """
        self.n = n
        self.data = NestD(data)

    def __recreate__(self,data={},n=None,*args,**kwargs):
        if n is None:
            n = self.n
        return self.__class__(data,n,*args,**kwargs)

    def __len__(self):
        return self.n

    def __getitem__(self,idx):
        return self.data.apply(lambda x:x[idx])

    def apply(self,func,*args,**kwargs):
        def _apply(x):
            if isinstance(x,Data):
                return x.apply(func,*args,**kwargs)
            else:
                return func(x,*args,**kwargs)
        return self.data.apply(_apply,*args,**kwargs)

    def __repr_header__(self):
        name = self.__class__.__name__
        return name + '{'

    def __repr__(self):
        return self.__repr_header__() + self.data.__repr__()

class SampledWithReplacement(Data):
    """
    E.g.
    >> import numpy as np
    >> data = SampledWithReplacement({'a':np.arange(20)},n=20)
    >> print data[np.arange(3,6)]
    
    """
    def __getitem__(self,idx):
        if isinstance(idx,slice):
            assert idx.stop is not None
            start, stop, step = 0, self.n, 1
            if idx.start is not None:
                if idx.start < 0:
                    start = self.n + idx.start
                else:
                    start = idx.start
            if idx.stop is not None:
                if idx.stop < 0:
                    stop = self.n + idx.stop
                else:
                    stop = idx.stop
            if idx.step is not None:
                step = idx.step
            idx = np.arange(self.n)[idx]
        if not isinstance(idx,collections.Hashable): 
            idx = np.random.randint(low=0,high=self.n,size=len(idx))
        return super(SampledWithReplacement,self).__getitem__(idx)


class SemiSupervisedDataTrain(object):

    def __init__(self,XL,YL,XU,YU=None,nL=None,nU=None,sample_func=None):
        self.data = NestD({'U':{'X':XU},
                           'L':{'X':XL,'Y':YL}})
        if YU is not None:
            self.data['U']['Y']=YU
        if nU is None:
            nU = len(XU)
        self.nU = nU
        if nL is None:
            nL = len(XL)
        self.nL = nL
        self.sample_func = sample_func

    def __len__(self):
        return self.nU

    def __getitem__(self,idx):
        if isinstance(idx,slice):
            idx = infer_slice(idx,self.nU)
        idx_U = idx
        idx_L = np.random.randint(low=0,high=self.nL,size=len(idx))
        U = self.data['U'].apply(lambda x:x[idx_U])
        L = self.data['L'].apply(lambda x:x[idx_L])
        rval = NestD({'U':U,'L':L})
        if self.sample_func:
            X = rval[:,['X']].apply(self.sample_func)
            Y = rval[:,['Y']]
            rval = X.updatepaths(*zip(*Y.walk()))
        return rval

    def __repr__(self):
        header = self.__class__.__name__ 
        subrepr = '\n  '.join(str(self.data.apply(np.shape)).split('\n'))
        return header + ': ' + subrepr


class SemiSupervisedDataEvaluate(object):

    def __init__(self,X,Y,n=None,sample_func=None):
        self.data = NestD({'U':{'X':X,'Y':Y},
                           'L':{'X':X,'Y':Y}})
        if n is None:
            n = len(X)
        self.n = n
        self.sample_func = sample_func

    def __len__(self):
        return self.n

    def __getitem__(self,idx):
        rval = self.data.apply(lambda x:x[idx])
        if self.sample_func:
            X = rval[:,['X']].apply(self.sample_func)
            Y = rval[:,['Y']]
            rval = X.updatepaths(*zip(*Y.walk()))
        return rval

    def __repr__(self):
        header = self.__class__.__name__ 
        subrepr = '\n  '.join(str(self.data.apply(np.shape)).split('\n'))
        return header + ': ' + subrepr


class SemiSupervisedMNIST(object):

    def __init__(self,labeled_per_class=10):
        """
        Divides training set into labeled and unlabeled data sets.  For valid and test sets,
        the unlabeled and labeled X are exactly the same.  SemiSupervisedMNIST() objects have
        self.train, self.valid, and self.test attributes, each of which is a DataNestD
        class.  To index one of these sets, use regular numpy slicing, e.g. self.train[idx]

        labeled_per_class: {int} default to 10, sets the number of labeled samples per digit 
                                 class in the training set.  E.g. labeled_per_class=10 will 
                                 result in 100 labeled training samples and 50000 unlabeled 
                                 training samples.
        """
        self.raw = loadDataset('mnist')
        self.nclasses = 10
        self.dim_observations = 784
        data = self.raw
        X = data['train']
        Y = data['train_y'].astype('int32')
        classes = range(self.nclasses)
        XL = []; YL = [];
        for c in classes:
            sel = Y == c
            nc = sel.sum()
            Xc = X[sel]
            Yc = Y[sel]
            idx = np.arange(nc)
            np.random.shuffle(idx)
            Xc = Xc[idx[:labeled_per_class]]
            Yc = Yc[idx[:labeled_per_class]]
            XL.append(Xc)
            YL.append(Yc)
        XL = np.vstack(XL)
        YL = np.hstack(YL)

        ntrainU = len(Y)
        ntrainL = len(YL)
        nvalid = len(data['valid_y'])
        ntest = len(data['test_y'])

        sample_func=lambda x: (x>=np.random.uniform(low=0,high=1,size=x.shape)).astype(float)
        self.train = SemiSupervisedDataTrain(
                        XU=X,
                        YU=Y,
                        XL=XL,
                        YL=YL,
                        sample_func=sample_func)
        self.valid = SemiSupervisedDataEvaluate(
                        X=data['valid'],
                        Y=data['valid_y'],
                        sample_func=sample_func)
        self.test = SemiSupervisedDataEvaluate(
                        X=data['test'],
                        Y=data['test_y'],
                        sample_func=sample_func)

        self.data = NestD({
            'train':self.train,
            'valid':self.valid,
            'test':self.test
        })

    def __repr__(self):
        header = self.__class__.__name__ 
        header += ' nclasses=%s, dim_observations=%s' % (self.nclasses,self.dim_observations)
        subrepr = '\n  '.join(str(self.data).split('\n'))
        return header + ': ' + subrepr

    def __getitem__(self,k):
        return self.data[k]

