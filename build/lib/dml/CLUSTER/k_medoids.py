from __future__ import division
import numpy as np
import scipy as sp
import pylab as py
import random
def EuclidDistance(x,y):
	return np.sqrt(np.sum(((np.array(x)-np.array(y))**2)))

class KmedoidsC:
	def __init__(self,X,K,dist=EuclidDistance):
		'''
			each row of X is a dot
			K is the number of class you want to cluster
			dis is a function
		'''
		self.dist=dist
		self.X=np.array(X)
		self.K=np.array(K)
		self.N=X.shape[0]
		self.labels = np.zeros(self.N, dtype=int)
		self.centroids=np.array(random.sample(X, K))
		self.J=self.calcJ()
	def calcJ(self):
		sum=0
		for i in range(self.N):
			sum += self.dist(self.X[i],self.centroids[self.labels[i]])
		return sum
	def distmat(self,X,Y):
		'''
		return the distance for centroids
		'''
		dm = np.zeros((X.shape[0],Y.shape[0]));
		for i in range(X.shape[0]):
			for j in range(Y.shape[0]):
				dm[i][j]=self.dist(X[i],Y[j])
		return dm
	def train(self,maxiter=100,treshhold=0.1):
		iter=0
		cJ=0
		while True:
			distmats = self.distmat(self.X,self.centroids)
			self.labels = distmats.argmin(axis=1)
			for j in range(self.K):
				idx_j = (self.labels == j).nonzero()
				distj = self.distmat(self.X[idx_j], self.X[idx_j])
				distsum = np.sum(distj, axis=1)
				icenter = distsum.argmin()
				self.centroids[j] = self.X[idx_j[0][icenter]]
			nJ=self.calcJ()
			cJ=self.J-nJ
			self.J=nJ
			if (cJ<treshhold and iter!=0):
				break
			iter+=1
			if (iter>maxiter):
				break
		pass
	def result(self):
		return self.centroids,self.labels,self.J
		