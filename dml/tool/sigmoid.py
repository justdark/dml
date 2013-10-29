import numpy as np
import scipy as sp
import warnings
#np.seterr(all='raise')
def sigmoid(x):
	#print x,'========='
	#print "======================="
	return 1/(1+np.float64(np.exp(-x)));
