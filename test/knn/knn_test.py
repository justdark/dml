from __future__ import division
import numpy as np
import scipy as sp
from dml.KNN.kd import KDTree
from dml.KNN  import KNNC
X=np.array([[2,5,9,4,8,7],[3,4,6,7,1,2]])
y=np.array([2,5,9,4,8,7])
knn=KNNC(X,1,y)
print knn.for_point([[2],[2]])