import numpy as np
import scipy as sp
from dml.SVM import SVMC
import matplotlib.pyplot as plt
X=[
[10.75,14.95],
[11.35,14.05],
[12,13.65],
[12.75,13.75],
[14.2,14.85],
[13.2,15.8],
[12.1,16.55],
[11.3,15.8],
[12.7,14.7],
[12.8,14.55],
[12.35,15.75],
[8.55,17.2],
[8.15,15.4],
[8.7,13.95],
[9.75,12.25],
[11.9,11.1],
[14.45,11.3],
[16.7,12.9],
[16.9,14.45],
[16.4,15.95],
[15,17.7],
[12.2,18.8],
[9.85,18.45],
[8.2,17.5],
[7.4,15.85],
[7.8,13.6],
[9.6,12.1],
[12.3,11.2],
[14.95,12.2],
[15.2,13.05],
[15.55,13.65],
[15.9,14.6],
[15.5,16.45],
[14.75,17.9],
[13.1,18.5],
[10.8,19.4],
[11.05,18.7]]


y=[
[0],
[0],
[0],
[0],
[0],
[0],
[0],
[0],
[0],
[0],
[0],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1]]
X=np.array(X).transpose()
print X.shape


y=np.array(y).flatten(1)
y[y==0]=-1
print y.shape
def Gauss_kernel(x,z,sigma=1):
		return np.exp(-np.sum((x-z)**2)/(2*sigma**2))
svms=SVMC(X,y,kernel=Gauss_kernel)
svms.train()
print len(svms.supportVec),"SupportVectors:"

for i in range(len(svms.supportVec)):
	t=svms.supportVec[i]
	print svms.X[:,t]
svms.error(X,y)
for i in range(svms.M):
			if  svms.y[i]==-1:
				plt.plot(svms.X[0,i],svms.X[1,i],'or')
			elif  svms.y[i]==1:
				plt.plot(svms.X[0,i],svms.X[1,i],'ob')
for i in svms.supportVec:
			plt.plot(svms.X[0,i],svms.X[1,i],'oy')
plt.show()
