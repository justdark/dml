from __future__ import division
import numpy as np
import scipy as sp
from operator import itemgetter
from scipy.spatial.distance import euclidean
from dml.tool import Heap
class KDNode:
	def __init__(self,x,y,l):
		self.x=x
		self.y=y
		self.l=l
		self.F=None
		self.Lc=None
		self.Rc=None
		self.distsToNode=None

class KDTree:
	def __init__(self,X,y=None,dist=euclidean):
		self.X=X
		self.k=X.shape[0] #N
		self.y=y
		self.dist=dist
		self.P=self.maketree(X,y,0)
		self.P.F=None
	def maketree(self,data,y,deep):
		if data.size==0:
			return None
		lenght = data.shape[0]
		case = data.shape[1]
		p=int((case)/2)
		l = (deep%self.k)
		#print data
		data=np.vstack((data,y))
		data=np.array(sorted(data.transpose(),key=itemgetter(l))).transpose()
		#print data
		y=data[lenght,:]
		data=data[:lenght,:]
		
		v=data[l,p]
		rP=KDNode(data[:,p],y[p],l)
		#print data[:,p],y[p],l
		if case>1:
			ldata=data[:,data[l,:]<v]
			ly=y[data[l,:]<v]
			data[l,p]=v-1
			rdata=data[:,data[l,:]>=v]
			ry=y[data[l,:]>=v]
			data[l,p]=v
			rP.Lc=self.maketree(ldata,ly,deep+1)
			if rP.Lc!=None:
				rP.Lc.F=rP
			rP.Rc=self.maketree(rdata,ry,deep+1)
			if rP.Rc!=None:
				rP.Rc.F=rP
		return rP

	def search_knn(self,P,x,k,maxiter=200):
		def pf_compare(a,b):
			return self.dist(x,a.x)<self.dist(x,b.x)
		def ans_compare(a,b):
			return self.dist(x,a.x)>self.dist(x,b.x)
		pf_seq=Heap(compare=pf_compare)
		pf_seq.insert(P)    #prior sequence
		ans=Heap(k,compare=ans_compare)  #ans sequence
		while pf_seq.counter>0:
			t=pf_seq.heap[1]
			pf_seq.delete(1)
			flag=True
			if ans.counter==k:
				now=t.F
				#print ans.heap[1].x,'========'
				if now != None:
					q=x.copy()
					q[now.l]=now.x[now.l]
					length=self.dist(q,x)
					if length>self.dist(ans.heap[1].x,x):
						flag=False
					else:
						flag=True
				else:
					flag=True
			if flag:
				tp,pf_seq,ans=self.to_leaf(t,x,pf_seq,ans)
			#print "============="
			#ans.insert(tp)
		return ans


	def to_leaf(self,P,x,pf_seq,ans):
		tp=P
		if tp!=None:
			ans.insert(tp)
			if tp.x[tp.l]>x[tp.l]:
				if tp.Rc!=None:
					pf_seq.insert(tp.Rc)
				if tp.Lc==None:
					return tp,pf_seq,ans
				else:
					return self.to_leaf(tp.Lc,x,pf_seq,ans)
			if tp.Lc!=None:
				pf_seq.insert(tp.Lc)
			if tp.Rc==None:
					return tp,pf_seq,ans
			else:
					return self.to_leaf(tp.Rc,x,pf_seq,ans)





