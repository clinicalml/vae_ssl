import numpy as np
from . import Namespace

class OutputLog(Namespace):
	def __init__(self,data={},axis=0,expand_dim=None,**kwargs):
		"""
		Essentially just a nested dictionary.  When new values (which are converted to arrays) or arrays are added to the log, if
		`expand_dim` is not None, then the array is first expanded along `expand_dim`, and then concatenated to the logged array
		along `axis`.
		"""
		super(OutputLog,self).__init__(data,**kwargs)
		self.axis=axis
		self.expand_dim=expand_dim
		self.log={}

	def add(self,outputs):
		"""
		concatenates self.log with arrays in leaves of outputs, store in self.log
		"""
		assert isinstance(outputs,dict) or isinstance(outputs,OutputLog), "outputs must be of type dict or OutputLog"
		if isinstance(outputs,OutputLog):
			outputs = outputs.log
		
		def _recursive_add(outputs,log):
			for k,v in outputs.iteritems():
				if isinstance(v,dict):
					if k not in log:
						log[k] = {}
					log[k] = _recursive_add(v,log[k])
				else:
					if v.size > 0:
						if v.size == 1:
							v = v.ravel()
						else:
							if expand_dim is not None:
								v = np.expand_dims(v,axis=expand_dim)
						if k not in log:
							log[k] = v
						else:
							log[k] = np.concatenate([log[k],v],axis=self.axis)
			return log
		self.log = _recursive_add(outputs,self.log)
		return self.log

	def apply(self,func,*args,**kwargs):
		"""
		apply func to all leaves in self.log
		"""
		def _recursive_apply(log,func,*args,**kwargs):
			for k,v in log.iteritems():
				if isinstance(v,dict):
					log[k] = _recursive_apply(v,func,*args,**kwargs)
				else:
					log[k] = func(v,*args,**kwargs)
			return log
		return _recursive_apply(self.log,func,*args,**kwargs)
		
		
