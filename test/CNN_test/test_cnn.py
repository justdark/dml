#coding=utf-8
from __future__ import division
import os
import numpy as np
from numpy import append, array, int8, uint8, zeros
import struct
from array import array as pyarray
from dml.CNN import CNNC,LayerC

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

    return images/256, labels

X,y=read(range(10))
print np.sum(X!=0)
test_x,test_y=read(range(10),dataset='testing')
test_groundTruth=np.zeros((10,test_y.size))
q=np.arange(0,test_y.size)
test_groundTruth[test_y.transpose(),q]=1
print test_groundTruth.argmax(axis=0)


groundTruth=np.zeros((10,y.size))
q=np.arange(0,y.size)
groundTruth[y.transpose(),q]=1
layers=[LayerC('i'),
		LayerC('c',out=6,kernelsize=5),
		LayerC('s',scale=2),
		LayerC('c',out=12,kernelsize=5),
		LayerC('s',scale=2)]
opts={}
opts['batchsize']=40
opts['numepochs']=1
opts['alpha']=1

a=CNNC(X,groundTruth,layers,opts)
#a.gradcheck(test_x[1:3,:,:],test_groundTruth[:,1:3])
a.train()
a.test(test_x,test_groundTruth)

