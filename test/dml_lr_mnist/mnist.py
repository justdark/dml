from __future__ import division
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

def read(digits, dataset = "training", path = "."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset is "training":
        fname_img = os.path.join(path, '../data/mnist/train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, '../data/mnist/train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, '../data/mnist/t10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, '../data/mnist/t10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in xrange(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in xrange(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

train_images, trian_labels = read(range(10), 'training')
test_images, test_labels=read(range(10),'testing')
theta = np.array([1,1,1,0,0,0])
#print train_images[1,:,:].reshape(1,-1)
num_train_case=train_images.shape[0]
num_test_case=test_images.shape[0]
train_images=train_images.reshape(num_train_case,-1)
test_images=test_images.reshape(num_test_case,-1)
print train_images.shape,trian_labels.shape
print test_images.shape,test_labels.shape
train_images=train_images/256;
test_images=test_images/256;
train_images=train_images.transpose()
test_images=test_images.transpose()
theta=np.ones((784,1))

a = LRC(train_images,trian_labels,nor=False)
theta=np.ones((10,784)).reshape(10*784)
np.set_printoptions(threshold='nan')
#print a.LRcost(theta)


a.train(200,True)
print test_labels[[156,522]]
tests=np.zeros((1,num_test_case)).reshape(1,-1);
tests[a.predict(test_images).reshape(1,-1)==test_labels.reshape(1,-1)]=1
print np.sum(tests)/num_test_case