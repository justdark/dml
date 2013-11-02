import numpy as np
import scipy as sp
import warnings
#np.seterr(all='raise')
def sign(x):
	#print x,'========='
	#print "======================="
	q=np.zeros(np.array(x).shape)
	q[x>=0]=1
	q[x<0]=-1
	return q
