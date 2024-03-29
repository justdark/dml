# -*- coding: utf8 -*-

from __future__ import division
import numpy as np
import scipy as sp
from dml.NB import NBC
'''
  Example of discreet data from  <统计学习方法>
  S=0
  M=1
  L=2
'''
X=np.array([
[1,0],
[1,1],
[1,1],
[1,0],
[1,0],
[2,0],
[2,1],
[2,1],
[2,2],
[2,2],
[3,2],
[3,1],
[3,1],
[3,2],
[3,2],
]).transpose()


y=np.array([
[-1],
[-1],
[1],
[1],
[-1],
[-1],
[-1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[-1],
]).transpose()

prop=np.ones((2,1))
a=NBC(X,y,prop)
a.train()
print a.pred([[2],
	[0]],showdetail=True)
'''
  Example of continuous feature
'''

X=np.array([
[6.2,11.8],
[7.3,11.45],
[9.85,11.3],
[11.15,9.45],
[9.8,7.75],
[7.7,7.3],
[6.7,8.35],
[7.85,9.65],
[9.5,9.3],
[7.55,8.4],
[6.1,9.4],
[6.05,10.7],
[7.3,9.7],
[16.4,16.2],
[18.15,14.35],
[20.45,14.35],
[22,15.15],
[22.35,17.5],
[21.55,18.85],
[18.35,19.45],
[16.6,17.45],
[18.4,15.85],
[21.45,16.55],
[20.75,18.25],
[18.75,17.9],
[19.25,16.95],
[20.1,16.3],
[20.4,17.15],
[19.35,17.7],
[18.35,16.85],
[18.95,15.55],
[20.55,15.25],
[21.45,16]]).transpose()


y=np.array([
[0],
[0],
[0],
[0],
[0],
[0],
[0],
[0],
[0],
[0],
[0],
[0],
[0],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1]]).transpose()

prop=np.zeros((2,1))
a=NBC(X,y,prop)
a.train()
print a.pred([[2,5,0,20,16],
	[0,5,0,20,14]],showdetail=True)