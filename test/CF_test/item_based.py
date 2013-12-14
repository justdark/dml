from __future__ import division
import numpy as np
import scipy as sp
from dml.CF import Item_based_C
def read_data():
	train=open('../data/movielens100k/u1.base').read().splitlines()
	test=open('../data/movielens100k/u1.test').read().splitlines()
	train_X=[]
	test_X=[]
	for line in train:
		p=line.split('	');
		train_X.append([int(p[0]),int(p[1]),int(p[2])])
	for line in test:
		p=line.split('	');
		test_X.append([int(p[0]),int(p[1]),int(p[2])])
	return train_X,test_X
train_X,test_X=read_data()
print np.array(train_X).shape,np.array(test_X).shape
a=Item_based_C(train_X)
a.test(test_X)