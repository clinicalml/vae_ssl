import collections
from theanomodels.datasets.load import loadDataset
import numpy as np
from models.namespace import Namespace

class DataNamespace(Namespace):

	def __init__(self,data={},n=None,*args,**kwargs):
		"""
		Converts data to a Namespace object.
		data: {dict,Namespace}
		n: {int}
		"""
		self.n = n
		super(DataNamespace,self).__init__(data,*args,**kwargs)

	def __recreate__(self,data={},n=None,*args,**kwargs):
		if n is None:
			n = self.n
		return self.__class__(data,n,*args,**kwargs)

	def __len__(self):
		return self.n

	def __getitem__(self,idx):
		if isinstance(idx,slice):
			idx = np.arange(self.n)[idx]
		elif isinstance(idx,tuple) and isinstance(idx[0],slice):
			assert False, 'cannot handle slicing along multiple axes'
		if isinstance(idx,collections.Hashable):
			return super(DataNamespace,self).__getitem__(idx)
		else:
			d = self.cascade(lambda x:x[idx])
			d.n = len(idx)
			return d

	def apply(self,*args,**kwargs):
		d = super(DataNamespace,self).apply(*args,**kwargs)
		return self.__recreate__(d,self.n,convert_children=False)

	def __repr_header__(self):
		return super(DataNamespace,self).__repr_header__() + ' n=%s' % self.n

	def __repr_leveled__(self,level=0):
		def repr_func(x):
			if isinstance(x,np.ndarray):
				return 'ndarray(shape=%s,dtype=%s)'%(str(x.shape),str(x.dtype))
			else:
				return x
		return super(DataNamespace,self.apply(repr_func)).__repr_leveled__(level)



class SampledWithReplacement(DataNamespace):
	"""
	E.g.
	>> import numpy as np
	>> data = SampledWithReplacement({'a':np.arange(20)},n=20)
	>> print data[np.arange(3,6)]
	
	"""
	def __getitem__(self,idx):
		if isinstance(idx,slice):
			idx = np.arange(self.n)[idx]
		if not isinstance(idx,collections.Hashable): 
			idx = np.random.randint(low=0,high=self.n,size=len(idx))
		return super(SampledWithReplacement,self).__getitem__(idx)



class SemiSupervisedMNIST(Namespace):

	def __init__(self,labeled_per_class=10):
		"""
		Divides training set into labeled and unlabeled data sets.  For valid and test sets,
		the unlabeled and labeled X are exactly the same.  SemiSupervisedMNIST() objects have
		self.train, self.valid, and self.test attributes, each of which is a DataNamespace
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

		self.train = DataNamespace({
				'U':{'X':X,'Y':Y},
				'L':SampledWithReplacement(
					{'X':XL,'Y':YL},n=ntrainL)
			},n=ntrainU)
		self.valid = DataNamespace({
				'U':{'X':data['valid'],'Y':data['valid_y']},
				'L':SampledWithReplacement({'X':data['valid'],'Y':data['valid_y']},n=nvalid)
			},n=nvalid)
		self.test = DataNamespace({
				'U':{'X':data['test'],'Y':data['test_y']},
				'L':SampledWithReplacement({'X':data['test'],'Y':data['test_y']},n=ntest)
			},n=ntest)

		data = {
			'train':self.train,
			'valid':self.valid,
			'test':self.test
		}
		super(SemiSupervisedMNIST,self).__init__(data,convert_children=False)


	def __repr_header__(self):
		header = ' nclasses=%s, dim_observations=%s' % (self.nclasses,self.dim_observations)
		return super(SemiSupervisedMNIST,self).__repr_header__() + header

