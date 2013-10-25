from __future__ import division
import csv
import numpy as np
def read_data():
	reader = csv.reader(file('../data/kaggle_mnist/train.csv', 'rb'))
	train_set=[]
	train_set_labels=[]
	i=0
	for line in reader:
		if (i!=0):
			line=map(int, line)
			train_set.append(line[1:])
			train_set_labels.append(line[0])
			#print np.array(train_set).shape
			#print numpy.array(line)
		i=i+1
	train_set=np.array(train_set)
	train_set_labels=np.array(train_set_labels)

	reader = csv.reader(file('../data/kaggle_mnist/test.csv', 'rb'))
	test_set=[]
	i=0
	for line in reader:
		if (i!=0):
			line=map(int, line)
			test_set.append(line)
			#print np.array(train_set).shape
			#print numpy.array(line)
		i=i+1
	test_set=np.array(test_set)
	return train_set,train_set_labels,test_set

train_set,train_set_labels,test_set=read_data()
train_set =train_set/256
test_set  =test_set/256
np.savez("data.npz",ts=train_set,tsl=train_set_labels,tests=test_set)




















