# coding: UTF-8
from __future__ import division
import numpy as np
import scipy as sp
from weakclassify import WEAKC
from dml.tool import sign
class ADABC:
	def __init__(self,X,y,Weaker=WEAKC):
		'''
		    X  is a  N*M matrix
			Weaker is a class of weak classifier
			It should have a 	train(self.W) method pass the weight parameter to train
								pred(test_set) method which return y formed by 1 or -1
			see detail in <统计学习方法>
		'''
		self.X=np.array(X)
		self.y=np.array(y).flatten(1)
		assert self.X.shape[1]==self.y.size
		self.Weaker=Weaker
		self.sums=np.zeros(self.y.shape)
		self.W=np.ones((self.X.shape[1],1)).flatten(1)/self.X.shape[1]
		self.Q=0
		#print self.W
	def train(self,M=4):
		'''
			M is the maximal Weaker classification
		'''
		self.G={}
		self.alpha={}
		for i in range(M):
			self.G.setdefault(i)
			self.alpha.setdefault(i)
		for i in range(M):
			self.G[i]=self.Weaker(self.X,self.y)
			e=self.G[i].train(self.W)
			#print self.G[i].t_val,self.G[i].t_b,e
			self.alpha[i]=float(1.0 / 2 * np.log((1 - e) / e))
			#print self.alpha[i]
			sg=self.G[i].pred(self.X)
			Z=self.W*np.exp(-self.alpha[i]*self.y*sg.transpose())
			self.W=(Z/Z.sum()).flatten(1)
			self.Q=i
			#print self.finalclassifer(i),'==========='
			if self.finalclassifer(i)==0:

				print i+1," weak classifier is enough to  make the error to 0"
				break
	def finalclassifer(self,t):
		'''
			the 1 to t weak classifer come together
		'''
		self.sums=self.sums+self.G[t].pred(self.X).flatten(1)*self.alpha[t]
		#print self.sums
		pre_y=sign(self.sums)
		#sums=np.zeros(self.y.shape)
		#for i in range(t+1):
		#	sums=sums+self.G[i].pred(self.X).flatten(1)*self.alpha[i]
		#	print sums
		#pre_y=sign(sums)
		t=(pre_y!=self.y).sum()
		return t
		
	def pred(self,test_set):
		test_set=np.array(test_set)
		assert test_set.shape[0]==self.X.shape[0]
		sums=np.zeros((test_set.shape[1],1)).flatten(1)

		for i in range(self.Q+1):
			sums=sums+self.G[i].pred(test_set).flatten(1)*self.alpha[i]
			#print sums
		pre_y=sign(sums)
		return pre_y