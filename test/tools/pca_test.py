from __future__ import division
import numpy as np
import scipy as sp
from scipy.io import loadmat
from dml.tool import pca,featurenormal,projectData,recoverData,displayData
import matplotlib.pyplot as plt
data = loadmat("../data/face/ex7data1.mat")
X = data['X'].transpose()
plt.axis([0,8,0,8])

'''
	simple test for PCA
'''

X_norm,mu,sigma=featurenormal(X)
plt.plot(X[0,:],X[1,:],'o')
#print mu,sigma,'======='
U,S = pca(X_norm)
#print U,S
tt=mu+U[:,0]
tp=mu+U[:,1]

plt.plot([mu[0],tt[0]] ,[mu[1],tt[1]])
plt.plot([mu[0],tp[0]],[mu[1],tp[1]])
plt.show()
plt.axis([-3,3,-3,3])

plt.plot(X_norm[0,:],X_norm[1,:],'o')

Z=projectData(X_norm, U, 1);
X_rec=recoverData(Z, U, 1);
plt.plot(X_rec[0,:],X_rec[1,:],'o')
for i in range(X.shape[1]):
	plt.plot([X_norm[0,i],X_rec[0,i]],[X_norm[1,i],X_rec[1,i]],'r')
plt.show()

'''
   face images dimension reduction
'''
data = loadmat("../data/face/ex7faces.mat")
X = data['X'].transpose()


X_norm,mu,sigma=featurenormal(X)
fig = plt.figure()
fig.add_subplot(1,2, 1)
plt.imshow(displayData(X_norm[:,:100]) , cmap='gray')

#PCA STEPs
[U, S] = pca(X_norm);
print S
K = 100;
Z = projectData(X_norm, U, K);
X_rec  = recoverData(Z, U, K);


fig.add_subplot(1,2, 2)
plt.imshow(displayData(X_rec[:,:100]) , cmap='gray')
plt.show()
