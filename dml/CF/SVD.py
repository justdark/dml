from __future__ import division
import numpy as np
import scipy as sp
from numpy.random import random
class  SVD_C:
	def __init__(self,X,k=20):
		'''
			k  is the length of vector
		'''
		self.X=np.array(X)
		self.k=k
		self.ave=np.mean(self.X[:,2])
		print "the input data size is ",self.X.shape
		self.bi={}
		self.bu={}
		self.qi={}
		self.pu={}
		self.movie_user={}
		self.user_movie={}
		for i in range(self.X.shape[0]):
			uid=self.X[i][0]
			mid=self.X[i][1]
			rat=self.X[i][2]
			self.movie_user.setdefault(mid,{})
			self.user_movie.setdefault(uid,{})
			self.movie_user[mid][uid]=rat
			self.user_movie[uid][mid]=rat
			self.bi.setdefault(mid,0)
			self.bu.setdefault(uid,0)
			self.qi.setdefault(mid,random((self.k,1))/10*(np.sqrt(self.k)))
			self.pu.setdefault(uid,random((self.k,1))/10*(np.sqrt(self.k)))
	def pred(self,uid,mid):
		self.bi.setdefault(mid,0)
		self.bu.setdefault(uid,0)
		self.qi.setdefault(mid,np.zeros((self.k,1)))
		self.pu.setdefault(uid,np.zeros((self.k,1)))
		if (self.qi[mid]==None):
			self.qi[mid]=np.zeros((self.k,1))
		if (self.pu[uid]==None):
			self.pu[uid]=np.zeros((self.k,1))
		ans=self.ave+self.bi[mid]+self.bu[uid]+np.sum(self.qi[mid]*self.pu[uid])
		if ans>5:
			return 5
		elif ans<1:
			return 1
		return ans
	def train(self,steps=20,gamma=0.04,Lambda=0.15):
		for step in range(steps):
			print 'the ',step,'-th  step is running'
			rmse_sum=0.0
			kk=np.random.permutation(self.X.shape[0])
			for j in range(self.X.shape[0]):
				i=kk[j]
				uid=self.X[i][0]
				mid=self.X[i][1]
				rat=self.X[i][2]
				eui=rat-self.pred(uid,mid)
				rmse_sum+=eui**2
				self.bu[uid]+=gamma*(eui-Lambda*self.bu[uid])
				self.bi[mid]+=gamma*(eui-Lambda*self.bi[mid])
				temp=self.qi[mid]
				self.qi[mid]+=gamma*(eui*self.pu[uid]-Lambda*self.qi[mid])
				self.pu[uid]+=gamma*(eui*temp-Lambda*self.pu[uid])
			gamma=gamma*0.93
			print "the rmse of this step on train data is ",np.sqrt(rmse_sum/self.X.shape[0])
			#self.test(test_data)
	def test(self,test_X):
		output=[]
		sums=0
		test_X=np.array(test_X)
		#print "the test data size is ",test_X.shape
		for i in range(test_X.shape[0]):
			pre=self.pred(test_X[i][0],test_X[i][1])
			output.append(pre)
			#print pre,test_X[i][2]
			sums+=(pre-test_X[i][2])**2
		rmse=np.sqrt(sums/test_X.shape[0])
		print "the rmse on test data is ",rmse
		return output