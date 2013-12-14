from __future__ import division
import numpy as np
import scipy as sp
class  Item_based_C:
	def __init__(self,X):
		self.X=np.array(X)
		print "the input data size is ",self.X.shape
		self.movie_user={}
		self.user_movie={}
		self.ave=np.mean(self.X[:,2])
		for i in range(self.X.shape[0]):
			uid=self.X[i][0]
			mid=self.X[i][1]
			rat=self.X[i][2]
			self.movie_user.setdefault(mid,{})
			self.user_movie.setdefault(uid,{})
			self.movie_user[mid][uid]=rat
			self.user_movie[uid][mid]=rat
		self.similarity={}
		pass
	def sim_cal(self,m1,m2):
		self.similarity.setdefault(m1,{})
		self.similarity.setdefault(m2,{})
		self.movie_user.setdefault(m1,{})
		self.movie_user.setdefault(m2,{})
		self.similarity[m1].setdefault(m2,-1)
		self.similarity[m2].setdefault(m1,-1)

		if self.similarity[m1][m2]!=-1:
			return self.similarity[m1][m2]
		si={}
		for user in self.movie_user[m1]:
			if user in self.movie_user[m2]:
				si[user]=1
		n=len(si)
		if (n==0):
			self.similarity[m1][m2]=1
			self.similarity[m2][m1]=1
			return 1
		s1=np.array([self.movie_user[m1][u] for u in si])
		s2=np.array([self.movie_user[m2][u] for u in si])
		sum1=np.sum(s1)
		sum2=np.sum(s2)
		sum1Sq=np.sum(s1**2)
		sum2Sq=np.sum(s2**2)
		pSum=np.sum(s1*s2)
		num=pSum-(sum1*sum2/n)
		den=np.sqrt((sum1Sq-sum1**2/n)*(sum2Sq-sum2**2/n))
		if den==0:
			self.similarity[m1][m2]=0
			self.similarity[m2][m1]=0
			return 0
		self.similarity[m1][m2]=num/den
		self.similarity[m2][m1]=num/den
		return num/den
	def pred(self,uid,mid):
		sim_accumulate=0.0
		rat_acc=0.0
		for item in self.user_movie[uid]:
			sim=self.sim_cal(item,mid)
			if sim<0:continue
			#print sim,self.user_movie[uid][item],sim*self.user_movie[uid][item]
			rat_acc+=sim*self.user_movie[uid][item]
			sim_accumulate+=sim
		#print rat_acc,sim_accumulate
		if sim_accumulate==0: #no same user rated,return average rates of the data
			return  self.ave
		return rat_acc/sim_accumulate
	def test(self,test_X):
		test_X=np.array(test_X)
		output=[]
		sums=0
		print "the test data size is ",test_X.shape
		for i in range(test_X.shape[0]):
			pre=self.pred(test_X[i][0],test_X[i][1])
			output.append(pre)
			#print pre,test_X[i][2]
			sums+=(pre-test_X[i][2])**2
		rmse=np.sqrt(sums/test_X.shape[0])
		print "the rmse on test data is ",rmse
		return output