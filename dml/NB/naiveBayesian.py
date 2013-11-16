from __future__ import division
import numpy as np
import scipy as sp
pi=3.14159
def gauss(a,mu,sigma):
	#print sigma
	return 1/(np.sqrt(2*pi))*np.exp(-(a-mu)**2/(2*sigma**2))
class NBC:
	def __init__(self,X,y,Indicator=None):
		'''
		 X 			 a N*M matrix where M is the train case number
		 			 should be number both continous and descret feature
		 y			 class label for classification
		 Indicator 	 show whether the feature is continous(0) or descret(1)
		             continous in default

		'''
		self.X=np.array(X)
		#print self.X.dtype
		self.N,self.M=self.X.shape
		self.y=np.array(y).flatten(1)
		assert self.X.shape[1]==self.y.size

		self.labels=np.unique(y)
		self.case={}
		self.mu={}
		self.sigma={}
		self.county={}
		if Indicator==None:
			self.Indicator=np.zeros((1,self.N)).flatten(1)
		else:
			self.Indicator=np.array(Indicator).flatten(1)
		assert self.Indicator.size==self.N
	def count(self,i,t,ys):
		counts=0
		for p in range(self.M):
			if (self.X[i,p]==t) and (self.y[p]==ys): 
				counts+=1
		return counts
	def train(self):
		for ys in self.labels:
			self.county.setdefault(ys)
			self.county[ys]=np.sum(self.y==ys)
		for i in range(self.N):
			if self.Indicator[i]==1:
				self.case.setdefault(i,{})
				self.case[i].setdefault('counts')
				self.case[i]['counts']=np.unique(self.X[i,:]).size
				for t in np.unique(self.X[i,:]):
					for ys in self.labels:
						self.case[i].setdefault(t,{})
						self.case[i][t].setdefault(ys)
						self.case[i][t][ys]=self.count(i,t,ys)
			elif self.Indicator[i]==0:
				self.mu.setdefault(i,{})
				self.sigma.setdefault(i,{})
				for ys in self.labels:
					tempx=self.X[i,self.y==ys]
					#print tempx
					self.sigma[i].setdefault(ys)
					self.sigma[i][ys]=np.std(tempx)
					self.mu[i].setdefault(ys)
					self.mu[i][ys]=np.mean(tempx)
					if self.sigma[i][ys]==0:
						self.sigma[i][ys]=1
	
	def nb_predict(self,x,showdetail=False):
		x=x.flatten(1)
		maxp=0
		y=self.labels[0]
		for ys in self.labels:
			now=(self.county[ys]+1)/(self.M+self.labels.size)
			for i in range(self.N):
				if self.Indicator[i]==1:
					self.case[i].setdefault(x[i],{})
					self.case[i][x[i]].setdefault(ys,0)
					now=now*((self.case[i][x[i]][ys]+1)/(self.county[ys]+self.case[i]['counts']))
				else:
					now=now*gauss(x[i],self.mu[i][ys],self.sigma[i][ys])
			if now>maxp:
				maxp=now
				y=ys
			if showdetail:
				print now,ys
		return y
	def pred(self,Test_X,showdetail=False):
		Test_X=np.array(Test_X)
		test_y=[]
		for i in range(Test_X.shape[1]):
			test_y.append(self.nb_predict(Test_X[:,i],showdetail))
		return test_y









