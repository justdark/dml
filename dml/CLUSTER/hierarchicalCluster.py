from __future__ import division
import numpy as np
import scipy as sp
import pylab as py
class HiPart:
	def __init__(self,x):
		'''
			x is a N*M matrix
		'''
		self.x=np.array(x)
		self.n=self.x.shape[0]
		self.m=self.x.shape[1]
		self.cen=np.mean(x,axis=1)

def HiCat(a,b):
	return HiPart(np.hstack((a.x,b.x)))

def HiLargest(a,b,measure):
	maxs=0
	for i in range(a.m):
		for j in range(b.m):
			ddist=measure(a.x[:,i],b.x[:,j])
			if maxs<ddist:
				maxs=ddist
	return maxs

def HiSmallest(a,b,measure):
	mins=1000000000
	for i in range(a.m):
		for j in range(b.m):
			ddist=measure(a.x[:,i],b.x[:,j])
			if mins>ddist:
				mins=ddist
	return mins

def HiAverage(a,b,measure):
	ddist=0
	q=0
	for i in range(a.m):
		for j in range(b.m):
			ddist=ddist+measure(a.x[:,i],b.x[:,j])
			q=q+1
	return ddist/q

def HiInversions(a,b,measure):
	return measure(a.cen,b.cen)

def EuclidDistance_dml_hiC(x,y):
	return np.sqrt(np.sum(((np.array(x)-np.array(y))**2)))

class HierarchicalClusterC:

	def __init__(self,X,option='complete',measure=None):
		if measure==None:
			self.measure=EuclidDistance_dml_hiC
		else:
			self.measure=measure

		if str(option)=='single':
			self.pdist=lambda a,b:HiSmallest(a,b,self.measure)
		elif str(option)=='complete':
			self.pdist=lambda a,b:HiLargest(a,b,self.measure)
		elif str(option)=='average':
			self.pdist=lambda a,b:HiAverage(a,b,self.measure)
		elif str(option)=='centroid':
			self.pdist=lambda a,b:HiInversions(a,b,self.measure)



		self.X=np.array(X)
		self.N=self.X.shape[0]
		self.M=self.X.shape[1]
		self.His={}
		self.His[0]=[]
		for i in range(self.X.shape[1]):
			self.His[0].append(HiPart(self.X[:,i].reshape(-1,1)))

	def train(self):
		for k in range(1,self.M):
			min_i=0
			min_j=0
			mins=1000000000
			for i in range(len(self.His[k-1])):
				for j in range(i+1,len(self.His[k-1])):
					dists=self.pdist(self.His[k-1][i],self.His[k-1][j])
					if dists<mins:
						mins=dists
						min_i=i
						min_j=j
			self.His[k]=[]
			for i in range(len(self.His[k-1])):
				if i not in [min_j,min_i]:
					self.His[k].append(self.His[k-1][i])
			self.His[k].append(HiCat(self.His[k-1][min_i],self.His[k-1][min_j]))

	def result(self,K=3):
		s=[]
		for i in range(len(self.His[self.M-K])):
			s.append(self.His[self.M-K][i].x)
		return s
		



