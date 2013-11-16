#coding:utf-8 
import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
from dml.KNN.kd import KDTree

#import pylab as py
class KNNC:
	"""docstring for KNNC"""
	def __init__(self,X,K,labels=None,dist=euclidean):
		'''
			X is a N*M matrix where M is the case 
			labels is prepare for the predict.
			dist is the similarity measurement way,

			The distance function can be ‘braycurtis’, ‘canberra’, 
			‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, 
			‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘kulsinski’, 
			‘mahalanobis’, 

		'''
		self.X = np.array(X)
		if labels==None:
			np.zeros((1,self.X.shape[1]))
		self.labels = np.array(labels)
		self.K = K
		self.dist = dist
		self.KDTrees=KDTree(X,labels,self.dist)

	def predict(self,x,k):
		ans=self.KDTrees.search_knn(self.KDTrees.P,x,k)
		dc={}
		maxx=0
		y=0
		for i in range(ans.counter+1):
			if i==0:
				continue
			dc.setdefault(ans.heap[i].y,0)
			dc[ans.heap[i].y]+=1
			if dc[ans.heap[i].y]>maxx:
				maxx=dc[ans.heap[i].y]
				y=ans.heap[i].y
		return y
	def for_point(self,test_x,k=None):
		if k==None:
			k=self.K
		ans=self.KDTrees.search_knn(self.KDTrees.P,np.array(test_x),k)
		result=[]
		for i in range(ans.counter+1):
			if i==0:
				continue
			result.append(ans.heap[i].x)
		return result
	def pred(self,test_x,k=None):
		'''
			test_x is a N*TM matrix,and indicate TM test case
			you can redecide the k
		'''
		if k==None:
			k=self.K
		test_case=np.array(test_x)
		y=[]
		for i in range(test_case.shape[1]):
			y.append(self.predict(test_case[:,i].transpose(),k))
		return y
