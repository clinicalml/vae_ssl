import collections

class NestD(dict,object):

    _valid_key_types = (str,unicode,int,float,bool)

    def __init__(self, data={}, convert_children=True, **kwargs):
        """
        Wrapper for dict with methods adapted for the hierarchical nature of dictionaries.
        data: {dict,list,iterator,NestD} if iterable and not of type dict, each element of iterable must have length 2
        convert_children: recursively convert all children in data to NestD types
        e.g.
        >> x = NestD({(1,1):{2:{3:4,5:6},'a':9}},convert_children=False)
        >> print x
        {
         (1, 1): {
           a: 9
           2: {
             3: 4
             5: 6
            }
          }
        }
        >> type(x[(1,1)])
        dict    
        >> y = NestD(x,convert_children=True)
        >> type(y[(1,1)])
        __main__.NestD
        """
        super(NestD,self).__init__(data,**kwargs)
        if convert_children:
            def _apply_func(d):
                if isinstance(d,dict) and not isinstance(d,NestD):
                    return self.__recreate__(d).apply(_apply_func)
                else:
                    return d
            temp = self.apply(_apply_func)
            for k in temp.keys():
                self[k] = temp[k]

    def __recreate__(self,*args,**kwargs):
        """
        use this to recreate myself
        """
        if 'convert_children' not in kwargs:
            kwargs['convert_children']=False
        return self.__class__(*args,**kwargs)

    def _check_if_valid_key_type(self,key):
        return any(map(lambda x: isinstance(key,x),self._valid_key_types))

    def __getitem__(self,x):
        def _get(d1,idx,idx_descendants=None):
            if self._check_if_valid_key_type(idx):
                d2 = super(NestD,d1).__getitem__(idx)
                if idx_descendants is not None and len(idx_descendants) > 0:
                    d2 = d2[idx_descendants]
            else: #idx is either slice or non-string iterable
                if isinstance(idx,slice):
                    if idx.start is None and idx.stop is None and idx.step is None:
                        key_filter = lambda a: True
                    else:
                        assert False, 'NestD get cannot accept numeric slices'
                elif isinstance(idx,collections.Iterable): #idx is a non-string iterable
                    key_filter = lambda a: a in idx
                else:
                    assert False, 'unhandled index type %s' % str(type(idx))
                d2 = d1.__recreate__()
                if idx_descendants is not None and len(idx_descendants) > 0:
                    for k in filter(key_filter,d1.keys()):
                        v = super(NestD,d1).__getitem__(k)
                        if isinstance(v,NestD):
                            d2[k] = v[idx_descendants]
                else:
                    for k in filter(key_filter,d1.keys()):
                        d2[k] = super(NestD,d1).__getitem__(k)
                d2 = d2.prune()
            return d2
        if isinstance(x,tuple):
            return _get(self,x[0],x[1:])
        else:
            return _get(self,x)

    def __setitem__(self, key, item):
        assert self._check_if_valid_key_type(key), '%s is not one of the valid key types: %s' % (type(key),str(self._valid_key_types))
        super(NestD,self).__setitem__(key,item)
        if isinstance(item, NestD):
            item._set_parent(self,key)

    def _set_parent(self,parent_obj,parent_key):
        self._parent_obj = parent_obj
        self._parent_key = parent_key

    def path(self):
        """
        if current namespace is the child of a parent namespace, return
        the keypath from the root namespace to the current namespace.
        e.g.
        >> x = NestD({1:{2:{3:4}}},convert_children=True)
        >> x[1][2].path()
        [1,2]
        """
        if hasattr(self,'_parent_key'):
            return self._parent_obj.path()+[self._parent_key]
        else:
            return []

    def __repr_header__(self):
        name = self.__class__.__name__
        return name + '{'

    def __repr_leveled__(self,level=0):
        indent='  '*level
        r = {}
        for k,v in self.iteritems():
            if isinstance(v,NestD):
                r[k] = v.__repr_leveled__(level+1)
            else:
                if hasattr(v,'__repr__'):
                    r[k] = v.__repr__()
                else:
                    r[k] = v
        header = self.__repr_header__()
        if len(r) > 0:
            return header+'\n'+('\n'.join('%s%s: %s' % (indent+' ',k,v) for k,v in r.iteritems()))+'}'
        else:
            return header+' }' 

    
    def __repr__(self):
        return self.__repr_leveled__()
        
    def walk(self):
        """
        returns an iterator that walks depth-first through namespace
        e.g.
        >> a = NestD({'(1,1)':{2:{3:4,5:'6'},'a':9}})
        >> print a
        {
         (1, 1): {
           a: 9
           2: {
             3: 4
             5: 6
            }
          }
        }    
        >> for k,v in a.walk(): print k,v
        ((1, 1), 'a') 9
        ((1, 1), 2, 3) 4
        ((1, 1), 2, 5) 6
        """
        def _walk(d,path=[]):
            for k in d.keys():
                _path=path+[k]
                if isinstance(d[k],dict):
                    for sub in _walk(d[k],_path):
                        yield sub
                else:
                    yield tuple(_path),d[k]
        for sub in _walk(self):
            yield sub

    def flatten(self,join=None):
        """
         converts hierarchical NestD to NestD of depth 1
         join: {str,None}, when join is not None, keypath elements are converted to string and joined by '/'
         e.g. NestD({'a':{'b':1}}) becomes NestD({'a/b':1})
        """
        if join:
            assert isinstance(join,str),'join must be a type of string'
            return self.__recreate__({join.join(k):v for k,v in self.apply(str,keys=True).walk()})
        else:
            return self.__recreate__(self.walk())
        
    def leaves(self,sort_keypaths=True):
        """
        returns all leaves in NestD flattened into a list
        """
        if sort_keypaths:
            return zip(*sorted(self.walk()))[1]
        else:
            return zip(*self.walk())[1]

    def updatepath(self,keypath,value):
        """
        e.g.
        >> a = NestD()
        >> a.updatepath([1,2,3],4)
        >> print a
        {
         1: {
           2: {
             3: 4
            }
          }
        }
        """
        k = keypath[0]
        if len(keypath) > 1:
            if k not in self.keys():
                self[k] = self.__recreate__()
            self[k].updatepath(keypath[1:],value)
        else:
            self[k] = value    
        return self

    def updatepaths(self,keypaths,values):
        for k,v in zip(keypaths,values):
            self.updatepath(k,v)
        return self

    def cascade(self,func,*args,**kwargs):
        """
        apply func to each NestD object in NestD hierarchy 
        with *args and **kwargs as parameters
        """
        d1 = self
        d2 = d1.__recreate__()
        for k in d1.keys():
            if isinstance(d1[k],NestD):
                d2[k] = d1[k].cascade(func,*args,**kwargs)
            else:
                d2[k] = d1[k]
        return func(d2,*args,**kwargs)
            
    def prune(self):
        """
        remove empty branches
        """
        def _prune(d1):
            d2 = d1.__recreate__()
            if len(d1)>0:
                for k in d1:
                    if isinstance(d1[k],NestD):
                        if len(d1[k])>0:
                            d2[k] = d1[k]
                    elif d1[k] is not None:
                        d2[k] = d1[k]
            return d2
        return self.cascade(_prune)
                    
    def to_dict(self):
        """
        convert all NestD nodes to dictionary
        """
        return self.cascade(dict)
            
    def apply(self,func,keys=False,*args,**kwargs):
        """
        recursively apply func to leaves or all keys of NestD
        keys: {True,False} defaults to False, if True, apply func to keys, else apply func to leaves
        *args, **kwargs: args and kwargs of func
        e.g.
        >> NestD({(1,1):{2:{3:4,5:6},'a':9}}).apply(lambda x: 2*x)
        {
         (1, 1): {
           a: 18
           2: {
             3: 8
             5: 12
            }
          }
        }
        >> NestD({(1,1):{2:{3:4,5:6},'a':9}}).apply(lambda x: str(x)+'!',keys=True)
        {
         (1, 1)!: {
           a!: 9
           2!: {
             3!: 4
             5!: 6
            }
          }
        }
        """
        d1 = self
        d2 = self.__recreate__(convert_children=False)
        for k1 in self.keys():
            if keys:
                k2 = func(k1)
            else:
                k2 = k1
            if isinstance(d1[k1],NestD):
                d2[k2] = d1[k1].apply(func,keys,*args,**kwargs)
            else:
                if keys:
                    d2[k2] = d1[k1]
                else:
                    d2[k2] = func(d1[k1],*args,**kwargs)
        return d2

