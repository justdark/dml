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
from dml.tool import normalize,disnormalize
from dml.NN import NNC
from dml.tool import sigmoid
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


groundTruth=np.zeros((10,train_images.shape[1]))
q=np.arange(0,train_images.shape[1])
groundTruth[trian_labels.transpose(),q]=1

architect = [784,500,10];
option ={}
a=NNC(architect,option)
#start=clock()
#for i in range(4):
#	a.test()
#finish=clock()
#print (finish-start)/10000
a.learningRate=1
a.weightPenaltyL2 = 0.0001
a.nonSparsityPenalty = 0.0001
a.dropoutFraction=0.1
a.inputZeroMaskedFraction=0.1
opts={'batchsize':100,'numepochs':10}
a.output='softmax'
a.train(train_images,groundTruth,opts)
result=a.nnpred(test_images)
writer = csv.writer(file('nnpredict.csv', 'wb'))
writer.writerow(['ImageId', 'Label'])
i=1
for p in result:
	writer.writerow([i,p])
	i=i+1
#print np.array(c[0:6]).reshape(2,3)