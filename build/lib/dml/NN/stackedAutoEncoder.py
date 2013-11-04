from __future__ import division
import numpy as np
import scipy as sp
from dml.NN import NNC
class SAEC:
	def __init__(self,architecture):
		self.architecture=architecture
		self.ae={}
		for i in range(len(architecture)-1):
			self.ae.setdefault(i)
			self.ae[i]=NNC([self.architecture[i],self.architecture[i+1],self.architecture[i]])
		pass
	def train(self,train_X,opts):
		x=train_X
		print "train for the sae"
		for i in range(len(self.architecture)-1):
			self.ae[i].train(x,x,opts)
			self.ae[i].nnff(x,x)
			x=self.ae[i].a[1]
			#print x.shape
			#print x[1:,:].shape
			x=x[1:,:]
		pass
		