import numpy as np
from namespace import Namespace
import numbers

class OutputLog(Namespace):
    def __init__(self,data={},axis=0,expand_dim=None,**kwargs):
        """
        Essentially just a nested dictionary.  When new values (which are converted to arrays) or arrays are added to the log, if
        `expand_dim` is not None, then the array is first expanded along `expand_dim`, and then concatenated to the logged array
        along `axis`.
        """
        self.axis=axis
        self.expand_dim=expand_dim
        super(OutputLog,self).__init__(data,**kwargs)
        if len(self)>0:
            preprocessed = self.apply(self._preprocess)
            for k in self.keys():
                self[k] = preprocessed[k]

    def __recreate__(self,data={},*args,**kwargs):
        return self.__class__(data,self.axis,self.expand_dim,*args,**kwargs)

    def _preprocess(self,x):
        if isinstance(x,np.ndarray):
            if x.size > 0:
                if x.size == 1:
                    x = x.ravel()
                elif self.expand_dim:
                    x = np.expand_dims(x,axis=self.expand_dim)
        elif isinstance(x,numbers.Number):
            x = np.asarray(x).ravel()
        return x



    def __repr_header__(self):
        return super(OutputLog,self).__repr_header__() + ' axis=%s, expand_dim=%s' % (self.axis,self.expand_dim)

    def __repr_leveled__(self,level=0):
        def repr_func(x):
            if isinstance(x,np.ndarray):
                return 'ndarray(shape=%s,dtype=%s)'%(str(x.shape),str(x.dtype))
            else:
                return x
        return super(OutputLog,self.apply(repr_func)).__repr_leveled__(level)

    def append(self,x):
        """
        concatenates self with arrays in leaves of x
        """
        assert isinstance(x,dict), "input must be a type of dict"
        
        def _recursive_add(x,log):
            for k,v in x.iteritems():
                if isinstance(v,dict):
                    if k not in log:
                        log[k] = self.__recreate__(convert_children=False)
                    log[k] = _recursive_add(v,log[k])
                else:
                    v = self._preprocess(v)
                    if v.size > 0:
                        if k not in log:
                            log[k] = v
                        else:
                            log[k] = np.concatenate([log[k],v],axis=self.axis)
            return log
        _recursive_add(x,self)
        return self

if __name__=='__main__':
    print OutputLog({'a':{},'b':{}})
    x = {'a':{'b':{'c':np.asarray(1.),
                   'd':np.random.randn(1),
                   'i':np.ones(0),
                   'h':1},
              'e':np.random.randn(5,5)},
          'f':{'g':np.random.randn(1,2,3)}}
    log = OutputLog()
    for i in range(3):
        log.append(x)
    print log
    log2 = OutputLog(axis=0,expand_dim=True)
    for i in range(3):
        log2.append(x)
    print log2
                  
        
        
