from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
def displayData(X,size=None,showit=False):
	'''
		X is a N*M  matrix
		size show each image's size,if None ,size will be sqrt(X.shape[0])
	'''
	x_length = int(np.sqrt(X.shape[1])); 
	y_length = int(np.sqrt(X.shape[1]));
	if size==None:
		size = np.sqrt(X.shape[0])
		size = [size,size]
	bigimg=np.zeros((x_length*size[0],y_length*size[1]));
	
	for i in range(x_length):
		for j in range(y_length):
			im = X[:,(i-1)*y_length+j].reshape(32,32).transpose()
			bigimg[i*32:(i+1)*32,(j)*32:(j+1)*32]=im
	if showit:
		fig = plt.figure()
		plt.imshow(bigimg , cmap='gray')
		plt.show()
	else:
		return bigimg
def showimage(X):
	fig = plt.figure()
	plt.imshow(X , cmap='gray')
	plt.show()