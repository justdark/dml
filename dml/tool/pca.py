from __future__ import division
import numpy as np
import scipy as sp
from scipy.linalg import svd
from dml.tool import normalize
def pca(X,axis=0):
	'''
		X is N*M matrix if axis is 0,otherwise is a M*N
		where M is traincase number,N is the number of features
		Returns the eigenvectors U, the eigenvalues (on diagonal) in S
	'''
	X_s=np.array(X)
	X_s = (X_s.transpose()-np.mean(X, axis=1)).transpose()
	if axis==0:
		N,M=X_s.shape
		Sigma=np.dot(X_s,X_s.transpose())/(M-1)
	else:
		M,N=X_s.shape
		Sigma=np.dot(X_s.transpose(),X_s)/(M-1)
	U,S,V = sp.linalg.svd(Sigma);
	return U,S

def projectData(X,U,K):
	#print X.shape
	#print U[:,:K].shape
	return np.dot(U[:,:K].transpose(),X)
def recoverData(Z,U,K):

	return np.dot(U[:,:K],Z)
