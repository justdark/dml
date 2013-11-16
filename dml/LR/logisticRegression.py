from __future__ import division
import numpy as np
import scipy as sp
import pylab as py
from scipy.optimize import minimize
from ..tool import sigmoid
from ..tool import normalize,normalize_by_extant
class LRC:
	''' Logistic Regression Class
		the class for Logistic Regression
		the parameter the user should offer is X and y
		X	-is a N*M matrix  
		y	-is a M  vector 
		lam	-is the parameter lambda for LR penalty
		actualy it's a softmax......=.=
		but they are same when the class_number is 2
		also see ufldl:
		http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression
	'''
	def __init__(self, X, y,lam=0.0001,nor=True):
		self.X = np.array(X)
		self.y = np.array(y).flatten(1)	
		(self.N,self.M)=X.shape
		self.nor=nor
		if (nor):
			self.X,self.scale,self.dvi= normalize(self.X)
		assert self.X.shape[1]==self.y.size
		self.lam = lam;
		self.label,self.y=np.unique(y,return_inverse=True)
		self.classNum =self.label.size
		self.theta = np.zeros((self.classNum,self.N)).reshape(self.classNum*self.N)
		self.groundTruth=np.zeros((self.classNum,self.M))
		self.groundTruth[self.y,np.arange(0,self.M)]=1
		if (y.shape[0] != self.M):
			print "the size of given data is wrong\n"
			return
			
	def LRcost(self,theta):
		#print self.X.shape,theta.reshape(self.classNum,self.N).shape
		theta=theta.reshape(self.classNum,self.N);
		M=np.dot(theta,self.X)
		#print theta.reshape(self.classNum,self.N)
		M=M-M.max()
		h=np.exp(M)
		h=np.true_divide(h,np.sum(h,0))
		#print -np.sum(groundTruth*np.log(h))/self.M
		cost = -np.sum(self.groundTruth*np.log(h))/self.M+self.lam/2.0*np.sum(theta**2);
		grad = -np.dot(self.groundTruth-h,self.X.transpose())/self.M+self.lam*theta;
		grad = grad.reshape(self.classNum*self.N)
		return cost,grad

	def train(self,maxiter=200,disp = False):
		#res,f,d=sp.optimize.fmin_l_bfgs_b(self.LRcost,self.theta,disp=1)
		x0=np.random.rand(self.classNum,self.N).reshape(self.classNum*self.N)/10
		res=sp.optimize.minimize(self.LRcost,x0, method='L-BFGS-B',jac=True,options={'disp': disp,'maxiter': maxiter})
		self.theta=res.x
		pass
		
	def predict(self,pred):
		if (self.nor):
			pred=normalize_by_extant(pred,self.scale,self.dvi)
		if (pred.shape[0]!=self.N):
			print "the data's size for predict is wrong\n"
			print "the process is stop"
			return 
		M=np.dot(self.theta.reshape(self.classNum,self.N),pred)
		h=np.exp(M)
		h=np.true_divide(h,np.sum(h,0)).transpose()
		h=h.argmax(axis=1)
		return self.label[h]
		
	def output(self):
		print np.dot(self.X,self.theta)