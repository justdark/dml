from __future__ import division
import csv
import numpy as np
import scipy as sp
import pylab as py
import struct
import os
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import matplotlib.pyplot as plt
from dml.LR import *
from dml.tool import normalize,disnormalize

r = np.load("../data/kaggle_mnist/data.npz")
train_images = r['ts']
trian_labels = r['tsl']
test_images  = r['tests']

#print train_images[1,:,:].reshape(1,-1)
num_train_case=train_images.shape[0]
num_test_case=test_images.shape[0]
print train_images.shape,trian_labels.shape
print test_images.shape

train_images=train_images.transpose()
test_images=test_images.transpose()
theta=np.ones((784,1))
a = LRC(train_images,trian_labels,nor=False)
theta=np.ones((10,784)).reshape(10*784)
np.set_printoptions(threshold='nan')
#print a.LRcost(theta)


a.train(200,True)
#print test_labels[[156,522]]
#tests=np.zeros((1,num_test_case)).reshape(1,-1);
result = a.predict(test_images)
writer = csv.writer(file('predict.csv', 'wb'))
writer.writerow(['ImageId', 'Label'])
i=1
for p in result:
	writer.writerow([i,p])
	i=i+1
#tests[a.predict(test_images).reshape(1,-1)==test_labels.reshape(1,-1)]=1
#print np.sum(tests)/num_test_case