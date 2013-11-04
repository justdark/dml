from __future__ import division
import numpy as np
import scipy as sp
import pylab as py
from scipy.cluster.vq import  kmeans2,whiten
'''
	the scipy contain a kmeans,so there is no need to write one
	bu for the convenience of using,I pack it with my code
	I don't know how to translate the input space to whitened space
	so If you need please add white
'''
class KMEANSC:
	def __init__(self,X,K):
		self.X=X
		self.K=K
		self.labels=[]
		self.centroids=[]
		
		pass
		
	def train(self,white=False):
		'''
			each train change everything
		'''
		if (white):
			self.centroids,self.labels=kmeans2(whiten(self.X),self.K,minit='random', missing='warn')
		else:
			self.centroids,self.labels=kmeans2(self.X,self.K,minit='random', missing='warn')
			
	def result(self):
		return self.centroids,self.labels
		
	def bfWhiteCen(self):
		''' if you use whiten on self.X in train,you need this to get the real controids
		'''
		Wcentroid=self.centroids
		print Wcentroid
		for i in range(self.K):
			Wcentroid[i]=np.sum(self.X[self.labels==i],axis=0)/list(self.labels).count(i)
		return Wcentroid