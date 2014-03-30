import numpy as np
import scipy as sp

#np.seterr(all='raise')
def expand(x,e):
	a=x
	for i in range(len(e)):
		a=np.repeat(a,e[i],axis=i)
	return a
