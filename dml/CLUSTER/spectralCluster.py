from __future__ import division
import numpy as np
import scipy as sp
from scipy.linalg import eigh,eig
from dml.CLUSTER.kmeans_iter import KMEANSC
def EuclidDistance(x,y):
	return np.sqrt(np.sum(((np.array(x)-np.array(y))**2)))

class SCC:
	def __init__(self,X,K,dist=EuclidDistance,ftype="Normalized"):
		'''
			X is a M*N matrix contain M case of train data
			K is the number of cluster you want to get
			dist is a function that to make the matrix
			ftype support "Normalized" or "Ratio"
			      two different way to calculate Laplacian
		'''
		self.X=X
		self.K=K
		self.dist=dist
		self.labels=[]
		self.centroids=[]
		self.W=self.distmat(X,X)
		self.D=np.diag(self.W.sum(axis=0))
		self.L=self.D-self.W
		self.ftype=ftype
		if ftype=="Normalized":
			self.D[self.D==0]=1
			self.L=self.D**(-0.5)*self.L*self.D**(-0.5)
		pass
	def train(self,maxiter=100,threshold=0.1):
		v,self.T=eig(self.L)
		#print v
		self.km=KMEANSC(self.T[:,1:self.K].transpose(),self.K)
		self.km.train(maxiter,threshold)
		self.labels=self.km.labels
	def distmat(self,X,Y):
		'''
		return the distance matrix for X and Y
		'''
		dm = np.zeros((X.shape[0],Y.shape[0]));
		for i in range(X.shape[0]):
			for j in range(Y.shape[0]):
				dm[i][j]=self.dist(X[i],Y[j])
		return dm
	def result(self):
		return self.labels