'''
	the whole structure of dml.NN is just like 
	DeepLearnToolbox(https://github.com/rasmusbergpalm/DeepLearnToolbox)
	I think the whole architecture of it is clear and easy to understand
	so I copy it to python
	I also recommand UFLDL(http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial)
	to learn Neural Network
	
	TODO:SAE,DAE and so on
'''
from __future__ import division
import numpy as np
import scipy as sp
import pylab as py
#from scipy.optimize import minimize
import datetime
from ..tool import sigmoid
from ..tool import normalize,normalize_by_extant
class NNC:
	def __init__(self, architecture,option={}):
		'''
			architecture is a list formed by sevral number indicate the NN shape
			for example:[784 200 10] shows a NN for mnist data with 
					one hidden layer of 200,
					input layer of 784
					output layer of 10
			ps: the bias element is not counted in these number
			
		'''
		self.testing=False
		self.architecture=np.array(architecture)
		self.learningRate=2
		self.momentum= 0.5
		self.output= 'sigm'
		self.activation ='sigm'
		self.weightPenaltyL2 = 0
		self.nonSparsityPenalty =0
		self.sparsityTarget=0.05
		self.inputZeroMaskedFraction=0
		self.dropoutFraction=0
		self.n=self.architecture.size
		self.W={}
		self.vW={}
		self.dW={}
		self.p={}
		self.a={}
		self.d={}
		self.dropOutMask={}
		for i in range(self.n):
			self.a.setdefault(i)
			
			self.d.setdefault(i)
			self.p.setdefault(i)
		
		for i in range(self.n-1):
			
			self.W.setdefault(i)
			self.vW.setdefault(i)
			
			self.p[i+1] = np.zeros((1,architecture[i+1]))
			self.W[i]=((np.random.rand(architecture[i]+1,architecture[i+1])-0.5)*2*4*np.sqrt(6/(architecture[i+1]+architecture[i]))); 
			#print architecture[i+1],architecture[i]
			self.vW[i] = np.zeros(self.W[i].shape)
			#self.W.append((np.random.rand(architecture[i+1],architecture[i]+1)-0.5)*8/np.sqrt(6/(architecture[i+1]+architecture[i])))
		
		
	def test(self):
		self.nncost(self.W)
		pass
		
	def handle_y_4classify(self,hy):
		groundTruth=np.zeros((self.architecture[self.n-1],hy.size))
		q=np.arange(0,hy.size)
		groundTruth[hy.transpose(),q]=1
		return groundTruth
		
	def nnbp(self):
		
		n = self.n
		
		sparsityError = 0
		if (self.output=='softmax' or self.output=='linear'):
			self.d[n-1] = -self.e
		elif self.output=='sigm':
			self.d[n-1] =-self.e*self.a[n-1]*(1-self.a[n-1])
		
		for i in range(n-2,0,-1):
			if (self.activation =='sigm'):
				d_act = self.a[i] * (1 - self.a[i])
			elif (self.activation =='tanh'):
				d_act = 1 - self.a[i]**2
			if (self.nonSparsityPenalty > 0):
				pi = np.tile(self.p[i], (self.a[i].shape[1], 1))
				#print pi,'============'
				#print np.zeros((self.a[i].shape[1],1)).shape,pi.shape
				sparsityError = np.concatenate((np.zeros((self.a[i].shape[1],1)),self.nonSparsityPenalty * (-self.sparsityTarget / pi + (1 - self.sparsityTarget) / (1 - pi))),axis=1).transpose()
				
			#print self.W[i].shape
			#print self.d[i + 1].shape,'sssss'
			
			if i+1==n-1:
				self.d[i] = (np.dot(self.W[i],self.d[i + 1] ) + sparsityError) * d_act;
			else:
				self.d[i] = (np.dot(self.W[i],self.d[i + 1][1:,:])  + sparsityError) * d_act;
			if(self.dropoutFraction>0):
				#print np.ones((self.d[i].shape[1],1)).shape
				#print self.dropOutMask[i].shape
				self.d[i] = self.d[i] * np.concatenate((np.ones((self.d[i].shape[1],1)).transpose() , self.dropOutMask[i]),axis=0)
		for i in range(n-1):
			#print self.a[i].shape

			if (i+1==n-1):
				
				self.dW[i] = np.dot(self.a[i],self.d[i + 1].transpose()) / self.d[i + 1].shape[1]
			else:
				self.dW[i] = np.dot(self.a[i],self.d[i + 1][1:,:].transpose()) /self.d[i + 1].shape[1]
			
			#print self.dW[i].shape,'ssssssssssssssssssss'
			
	def nnff(self,X,y=None):
		'''
			X is a matrix with shape N*M ,M is the number of train_case,N is the input size 784
			y is the labels
			W is a dictionary
		'''
		
		n = self.n;
		M = X.shape[1];
		X = np.concatenate((np.ones((1,M)),X),axis=0)
		self.a[0]=X;
		for i in range(n-1):
			if i==0:
				continue
			#print self.a[i-1].shape
			#print np.dot(self.W[i-1].transpose(),self.a[i-1])
			if (self.activation =='sigm'):
				self.a[i]=sigmoid(np.dot(self.W[i-1].transpose(),self.a[i-1]))
			elif (self.activation=='tanh'):
				self.a[i]=np.tanh(np.dot(self.W[i-1].transpose(),self.a[i-1]))
			
			
			if (self.dropoutFraction>0):
				if (self.testing):
					self.a[i]=self.a[i]*(1-self.dropoutFraction)
				else:
					self.dropOutMask.setdefault(i)
					self.dropOutMask[i]=(np.random.rand(self.a[i].shape[0],self.a[i].shape[1])>self.dropoutFraction)
					
					
					self.a[i] = self.a[i]*self.dropOutMask[i];
			if (self.nonSparsityPenalty>0):
				self.p[i] = 0.8 * self.p[i] + 0.2*np.mean(self.a[i],axis=1);
				#self.p[i] =np.mean(self.a[i],axis=1)


			self.a[i]=np.concatenate((np.ones((1,M)),self.a[i]),axis=0)
		
		#print self.a[n-1]
		#set the output
		#settle softmax
		#print self.W[n - 2].transpose()
		if (self.output=='softmax'):
			self.a[n-1] = np.dot(self.W[n - 2].transpose(),self.a[n - 2])
			self.a[n-1]=np.exp(self.a[n-1]-self.a[n-1].max())
			self.a[n-1]=np.true_divide(self.a[n-1],np.sum(self.a[n-1],0))
		elif (self.output=='linear'):
			self.a[n-1] = np.dot(self.W[n - 2].transpose(),self.a[n - 2])
		elif (self.output=='sigm'):
			self.a[n-1] = sigmoid(np.dot(self.W[n - 2].transpose(),self.a[n - 2]))
		if (y!=None):
		
			self.e= y-self.a[n-1]
			if (self.output=='sigm' or self.output=='linear'):
				self.L = 1/2*(self.e**2).sum() / M; 
			elif  (self.output=='softmax'):
				self.L = (-y*np.log(self.a[n-1])).sum() / M;
		#print self.L
	def train(self,train_X,train_y,opts):
		
		'''
		  train_X is a matrix with shape N*M ,M is the number of train_case,N is the input size,eg: 784 in MNIST data
		'''
		m = train_X.shape[1]
		batchsize = opts['batchsize']
		numepochs = opts['numepochs']
		numbatches = int(m / batchsize)
		kk=np.random.permutation(m)
		for i in range(numepochs):
			starttime = datetime.datetime.now()
			print 'the ',i,' th epochs is running'
			for j in range(numbatches):
				batch_x=train_X[:,kk[(j)*batchsize:(j+1)*batchsize]].copy();
				if(self.inputZeroMaskedFraction != 0):
					batch_x = batch_x*(np.random.rand(batch_x.shape[0],batch_x.shape[1])>self.inputZeroMaskedFraction) 
				batch_y=train_y[:,kk[(j)*batchsize:(j+1)*batchsize]].copy()
				
				self.nnff(batch_x, batch_y);
				self.nnbp();
				self.nnapplygrads();
			endtime = datetime.datetime.now()
			print 'cost ',(endtime - starttime).seconds,'seconds\n'
	def nnapplygrads(self):
		for i in range(self.n-1):
			if(self.weightPenaltyL2>0):
				#print np.zeros((self.W[i].shape[1],1)).shape,self.W[i][1:,:].shape
				#print self.W[i]
				#print self.dW[i].shape,np.concatenate((np.zeros((self.W[i].shape[1],1)).transpose(),self.W[i][1:,:]),axis=0).shape
				##print self.W[i]
				#dsaaaaaaaaaaaaaaaaa
				dW = self.dW[i]+self.weightPenaltyL2*self.W[i]
				#dW = self.dW[i] + self.weightPenaltyL2 *np.concatenate((np.zeros((self.W[i].shape[1],1)).transpose(),self.W[i][1:,:]),axis=0) 
			else:
				dW = self.dW[i];
        
			dW = self.learningRate * dW;
			
			if(self.momentum>0):
				self.vW[i] = self.momentum*self.vW[i] + dW;
				dW = self.vW[i];
        
			self.W[i] = self.W[i] - dW;
		pass
		
	def nnpred(self,test_X):
		self.testing=True;
		nn = self.nnff(test_X, np.zeros((test_X.shape[1], self.architecture[self.n-1])).transpose());
		self.testing=False;
		print self.a[self.n-1].shape
		return  self.a[self.n-1].argmax(axis=0)

	