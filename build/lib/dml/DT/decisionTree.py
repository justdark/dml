from __future__ import division
import numpy as np
import scipy as sp
import pylab as py
def pGini(y):
		ty=y.reshape(-1,).tolist()
		label = set(ty)
		sum=0
		num_case=y.shape[0]
		#print y
		for i in label:
			sum+=(np.count_nonzero(y==i)/num_case)**2
		return 1-sum
	
class DTC:
	def __init__(self,X,y,property=None):
		'''
			this is the class of Decision Tree
			X is a M*N array where M stands for the training case number
								   N is the number of features
			y is a M*1 vector
			property is a binary vector of size N
				property[i]==0 means the the i-th feature is discrete feature,otherwise it's continuous
				in default,all feature is discrete
				
		'''
		'''
			I meet some problem here,because the ndarry can only have one type
			so If your X have some string parameter,all thing will translate to string
			in this situation,you can't have continuous parameter
			so remember:
			if you have continous parameter,DON'T PUT any STRING IN X  !!!!!!!!
		'''
		self.X=np.array(X)
		self.y=np.array(y)
		self.feature_dict={}
		self.labels,self.y=np.unique(y,return_inverse=True)
		self.DT=list()
		if (property==None):
			self.property=np.zeros((self.X.shape[1],1))
		else:
			self.property=property
			
		for i in range(self.X.shape[1]):
			self.feature_dict.setdefault(i)
			self.feature_dict[i]=np.unique(X[:,i])

		if (X.shape[0] != y.shape[0] ):
			print "the shape of X and y is not right"
			
		for i in range(self.X.shape[1]):
			for j in self.feature_dict[i]:
				pass#print self.Gini(X,y,i,j)
		pass

	def Gini(self,X,y,k,k_v):
		if (self.property[k]==0):
			#print X[X[:,k]==k_v],'dasasdasdasd'
			#print X[:,k]!=k_v
			c1 = (X[X[:,k]==k_v]).shape[0]
			c2 = (X[X[:,k]!=k_v]).shape[0]
			D = y.shape[0]
			return c1*pGini(y[X[:,k]==k_v])/D+c2*pGini(y[X[:,k]!=k_v])/D
		else:
			c1 = (X[X[:,k]>=k_v]).shape[0]
			c2 = (X[X[:,k]<k_v]).shape[0]
			D = y.shape[0]
			#print c1,c2,D
			return c1*pGini(y[X[:,k]>=k_v])/D+c2*pGini(y[X[:,k]<k_v])/D
		pass
	def makeTree(self,X,y):
		min=10000.0
		m_i,m_j=0,0
		if (np.unique(y).size<=1):

			return (self.labels[y[0]])
		for i in range(self.X.shape[1]):
			for j in self.feature_dict[i]:
				p=self.Gini(X,y,i,j)
				if (p<min):
					min=p
					m_i,m_j=i,j
		
		

		if (min==1):
			return (y[0])
		left=[]
		righy=[]
		if (self.property[m_i]==0):
			left = self.makeTree(X[X[:,m_i]==m_j],y[X[:,m_i]==m_j])
			right = self.makeTree(X[X[:,m_i]!=m_j],y[X[:,m_i]!=m_j])
		else :
			left = self.makeTree(X[X[:,m_i]>=m_j],y[X[:,m_i]>=m_j])
			right = self.makeTree(X[X[:,m_i]<m_j],y[X[:,m_i]<m_j])
		return [(m_i,m_j),left,right]
	def train(self):
		self.DT=self.makeTree(self.X,self.y)
		print self.DT
		
	def pred(self,X):
		X=np.array(X)
		  
		result = np.zeros((X.shape[0],1))
		for i in range(X.shape[0]):
			tp=self.DT
			while ( type(tp) is  list):
				a,b=tp[0]
				
				if (self.property[a]==0):
					if (X[i][a]==b):
						tp=tp[1]
					else:
						tp=tp[2]
				else:
					if (X[i][a]>=b):
						tp=tp[1]
					else:
						tp=tp[2]
			result[i]=self.labels[tp]
		return result
		pass
	