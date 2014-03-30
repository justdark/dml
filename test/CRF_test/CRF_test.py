from __future__ import division
import os
import numpy as np
from dml.CRF import CRFC
'''
	data from conll2000 NP chunking:http://www.cnts.ua.ac.be/conll2000/chunking/
'''
train_file=open('../data/crf_data/train.txt','r')
test_file=open('../data/crf_data/test.txt','r')

model=CRFC(train_file)