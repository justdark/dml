from __future__ import division
import numpy as np
import scipy as sp
__all__ = [
'normalize',
'disnormalize',
'normalize_by_extant',
'featurenormal'
]
def featurenormal(X,axis=0):
	'''
	   X is N*M
	   axis==0: columns
	   axis==1: rows
	'''
	mu=np.array(X).mean(not axis)
	X_norm=X-mu.reshape(X.shape[0],-1)
	sigma=np.std(X_norm,axis=not axis)
	X_norm=X_norm/sigma.reshape(X.shape[0],-1)

	return X_norm,mu,sigma

def normalize(X,Ub=0,Lb=1,order=1):
	'''
	normalize the data
	Ub and Lb is Upper and lower bounds 
	order = 0 normalized by all elements
		  =	1 normalized by rows
		  = 2 normalized by columns
	return :
		data: normalized data
		scale,divd: to recover thedata
	'''
	MAX=0
	MIN=0
	X=np.array(X)
	if (order==0):
		MAX,MIN=X.max(),X.min()
	elif (order==1):
		MAX,MIN=X.max(1),X.min(1)
	else:
		MAX,MIN=X.max(0),X.min(0)
	
	scale,divd	= (MIN,MAX-MIN)
	if order!=0:
		scale[divd==0]	= 0
		divd[divd==0]	= MAX[divd==0]
	if (order==0):
		X=(X-scale)/divd*(Lb-Ub)-Ub
	elif (order==1):
		X=(X-scale.reshape(-1,1))/divd.reshape(-1,1)*(Lb-Ub)-Ub
	else:
		X=(X-scale.reshape(1,-1))/divd.reshape(1,-1)*(Lb-Ub)-Ub
	return X,scale,divd

def disnormalize(X,scale,divd,Ub=0,Lb=1,order=1):
	if (order==0):
		X=(X+Ub)/(Lb-Ub)*divd+scale
	elif (order==1):
		X=(X+Ub)/(Lb-Ub)*divd.reshape(-1,1)+scale.reshape(-1,1)
	else:
		X=(X+Ub)/(Lb-Ub)*divd.reshape(1,-1)+scale.reshape(1,-1)
	return X
def normalize_by_extant(X,scale,divd,Ub=0,Lb=1,order=1):
	if (order==0):
		X=(X-scale)/divd*(Lb-Ub)-Ub
	elif (order==1):
		X=(X-scale.reshape(-1,1))/divd.reshape(-1,1)*(Lb-Ub)-Ub
	else:
		X=(X-scale.reshape(1,-1))/divd.reshape(1,-1)*(Lb-Ub)-Ub
	return X