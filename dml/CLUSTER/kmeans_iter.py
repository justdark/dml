from __future__ import division
import numpy as np
import scipy as sp
import pylab as py
import random
from scipy.cluster.vq import  kmeans2,whiten
'''
	the scipy contain a kmeans,so there is no need to write one
	bu for the convenience of using,I pack it with my code
	I don't know how to translate the input space to whitened space
	so If you need please add white
	reference to http://blog.pluskid.org/?p=17
'''
class KMEANSC:
	def __init__(self,X,K):
		self.X=np.array(X)
		self.K=np.array(K)
		self.N=X.shape[0]
		self.labels = np.zeros(self.N, dtype=int)
		self.centroids=np.array(random.sample(X, K))
		self.J=self.calcJ()
		pass
	def calcJ(self):
		sum=0
		for i in range(self.N):
			sum += np.sum((self.X[i]-self.centroids[self.labels[i]])**2)
		return sum	
	def distmat(self,X,Y):
		'''
		return the distance for centroids
		'''

		dm = np.zeros((X.shape[0],Y.shape[0]));
		for i in range(X.shape[0]):
			for j in range(Y.shape[0]):
				dm[i][j]=np.sum(((X[i]-Y[j])**2))
		return dm
		
	def train(self,maxiter=100,threshold=0.1):
		'''
			each train change everything
		'''
		iter=0
		cJ=0
		while True:
			distmats = self.distmat(self.X,self.centroids)
			self.labels = distmats.argmin(axis=1)
			for j in range(self.K):
				idx_j = (self.labels == j).nonzero()
				self.centroids[j] = self.X[idx_j].mean(axis=0)
			nJ=self.calcJ()
			cJ=self.J-nJ
			self.J=nJ
			if (cJ<threshold and iter!=0):
				break
			iter+=1
			if (iter>maxiter):
				break
			
	def result(self):
		return self.centroids,self.labels,self.J
		