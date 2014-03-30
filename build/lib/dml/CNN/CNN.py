from __future__ import division
import numpy as np
import scipy as sp
from scipy.signal import convolve as conv
from dml.tool import sigmoid,expand,showimage
from numpy import rot90
'''
	this algorithm have refered to the DeepLearnToolBox(https://github.com/rasmusbergpalm/DeepLearnToolbox)
	also:[1]:"Notes on Convolutional Neural Networks" Jake Bouvrie 2006 - How to implement CNNs
	I want to implement as [1] described,where the subsampling layer have sigmoid function
	 but finally it does not converge,but I can pass the gradcheck!!
	 (this version is dml/CNN/cnn.py.old ,if you can figure out what is wrong in the code,PLEASE LET ME KNOW)
	at last I changed code back to simple version,delete the sigmoid in 's' layer  
	ps:this code in python is too slow!don't use it do anything except reading.
'''
class LayerC:
	def __init__(self,types='i',out=0,scale=0,kernelsize=0):
		self.types=types
		self.a=None
		self.b=None
		self.d=None
		if (types=='i'):
			pass
		elif (types=='c'):
			self.out=out
			self.kernelsize=kernelsize
			self.k=None
		elif (types=='s'):
			self.scale=scale
			self.Beta={}
			self.dBeta={}

class CNNC:
	def __init__(self,X,y,layers,opts):
		self.X=np.array(X)
		self.y=np.array(y)
		self.layers=layers
		self.opts=opts
		inputmap = 1
		mapsize = np.array(self.X[0].shape)
		
		for i in range(len(self.layers)):
			if self.layers[i].types=='s':
				mapsize = mapsize / self.layers[i].scale
				assert np.sum(np.floor(mapsize)== mapsize)==mapsize.size
				self.layers[i].b={}
				self.layers[i].db={}
				for j in range(inputmap):
					self.layers[i].b.setdefault(j,0)
					self.layers[i].db.setdefault(j,0)
					self.layers[i].Beta.setdefault(j,1)
					self.layers[i].dBeta.setdefault(j,0.0)
				pass
			if self.layers[i].types=='c':
				mapsize = mapsize - self.layers[i].kernelsize + 1
				fan_out = self.layers[i].out*self.layers[i].kernelsize**2 
				self.layers[i].k={}
				self.layers[i].dk={}  
				self.layers[i].b={}
				self.layers[i].db={}
				for j in range(self.layers[i].out):
					
					fan_in = inputmap*self.layers[i].kernelsize**2
					for t in range(inputmap):
						self.layers[i].k.setdefault(t,{})
						self.layers[i].k[t].setdefault(j)
						self.layers[i].k[t][j]=(np.random.rand(self.layers[i].kernelsize,self.layers[i].kernelsize)-\
												0.5)*2*np.sqrt(6/(fan_out+fan_in))
						self.layers[i].dk.setdefault(t,{})
						self.layers[i].dk[t].setdefault(j)
						self.layers[i].dk[t][j]=np.zeros(self.layers[i].k[t][j].shape)
					self.layers[i].b.setdefault(j,0)
					self.layers[i].db.setdefault(j,0)	
				inputmap=self.layers[i].out
			if self.layers[i].types=='i':
				pass
		fvnum = np.prod(mapsize)*inputmap; 
		onum = self.y.shape[0];
		self.ffb=np.zeros((onum,1))
		self.ffW=(np.random.rand(onum, fvnum)-0.5)*2*np.sqrt(6/(onum+fvnum))
	def cnnff(self,x):
		#print x
		self.layers[0].a={}
		self.layers[0].a.setdefault(0)
		self.layers[0].a[0]=x.copy()
		inputmap=1
		n=len(self.layers)
		
		for l in range(1,n):
			if self.layers[l].types=='s':
				for j in range(inputmap):
				
					temp=np.ones((self.layers[l].scale,self.layers[l].scale))/(self.layers[l].scale**2)
					z=conv(self.layers[l-1].a[j],np.array([temp]), 'valid')
					z=np.array(z)[:,::self.layers[l].scale,::self.layers[l].scale]

					if self.layers[l].a==None:
						self.layers[l].a={}
					self.layers[l].a.setdefault(j)
					self.layers[l].a[j] =z
					

			if self.layers[l].types=='c':
				if self.layers[l].a==None:
					self.layers[l].a={}
				for j in range(self.layers[l].out): #for each outmaps
					z = np.zeros(self.layers[l-1].a[0].shape - np.array([0,self.layers[l].kernelsize-1,self.layers[l].kernelsize-1]))
					for i in range(inputmap):       #cumulate from inputmaps
						z+=conv(self.layers[l-1].a[i],np.array([self.layers[l].k[i][j]]),'valid')			
					self.layers[l].a.setdefault(j)
					self.layers[l].a[j]=sigmoid(z+self.layers[l].b[j])
				inputmap = self.layers[l].out
				
		self.fv=None
		for j in range(len(self.layers[n-1].a)):
			sa=self.layers[n-1].a[j].shape
			p=self.layers[n-1].a[j].reshape(sa[0],sa[1]*sa[2]).copy()
			if (self.fv==None):
				self.fv=p
			else:
				self.fv=np.concatenate((self.fv,p),axis=1)
		self.fv=self.fv.transpose()
		self.o=sigmoid(np.dot(self.ffW,self.fv) + self.ffb)

		

	def cnnbp(self,y):
		n=len(self.layers)
		self.e=self.o-y
		self.L=0.5*np.sum(self.e**2)/self.e.shape[1]
		self.od=self.e*(self.o*(1-self.o))
		
		self.fvd=np.dot(self.ffW.transpose(),self.od)
		if self.layers[n-1].types=='c':
			self.fvd=self.fvd*(self.fv*(1-self.fv))
		sa=self.layers[n-1].a[0].shape
		fvnum=sa[1]*sa[2]
		for j in range(len(self.layers[n-1].a)):
			if self.layers[n-1].d==None:
				self.layers[n-1].d={}
			self.layers[n-1].d.setdefault(j)
			self.layers[n-1].d[j]=self.fvd[(j*fvnum):((j+1)*fvnum),:].transpose().reshape(sa[0],sa[1],sa[2])

		for l in range(n-2,-1,-1):
			if self.layers[l].types=='c':
				for j in range(len(self.layers[l].a)):
					if self.layers[l].d==None:
						self.layers[l].d={}
					self.layers[l].d.setdefault(j)
					self.layers[l].d[j]=self.layers[l].a[j]*(1-self.layers[l].a[j])*\
										np.kron(self.layers[l+1].d[j],np.ones((	self.layers[l+1].scale,self.layers[l+1].scale))/(self.layers[l+1].scale**2))
					
					
			elif self.layers[l].types=='s':
				for j in range(len(self.layers[l].a)):
					if self.layers[l].d==None:
						self.layers[l].d={}
					self.layers[l].d.setdefault(j)
					z=np.zeros(self.layers[l].a[0].shape)
					for i in range(len(self.layers[l+1].a)):
						rotated=np.array([rot90(self.layers[l+1].k[j][i],2)])
						z=z+conv(self.layers[l+1].d[i],rotated,'full')
					self.layers[l].d[j]=z

		for l in range(1,n):
			m=self.layers[l].d[0].shape[0]
			if self.layers[l].types=='c':
				for j in range(len(self.layers[l].a)):
					for i in range(len(self.layers[l-1].a)):
						#self.layers[l].dk[i][j]=rot90(conv(self.layers[l-1].a[i],rot90(self.layers[l].d[j],2),'valid'),2)
						self.layers[l].dk[i][j]=self.layers[l].dk[i][j]*0
						for t in range(self.layers[l].d[0].shape[0]):
							self.layers[l].dk[i][j]+=rot90(conv(self.layers[l-1].a[i][t],rot90(self.layers[l].d[j][t],2),'valid'),2)
						
						self.layers[l].dk[i][j]=self.layers[l].dk[i][j]/m
					self.layers[l].db[j]=np.sum(self.layers[l].d[j])/m
		self.dffW=np.dot(self.od,self.fv.transpose())/self.od.shape[1]
		self.dffb = np.mean(self.od,1).reshape(self.ffb.shape);


	def cnnapplygrads(self,alpha=0.1):
		n=len(self.layers)
		for l in range(1,n):
			if self.layers[l].types=='c':
				for j in range(len(self.layers[l].a)):
					for i in range(len(self.layers[l-1].a)):
						self.layers[l].k[i][j]-=alpha*self.layers[l].dk[i][j]
					self.layers[l].b[j]-=alpha*self.layers[l].db[j]
				pass
		
		self.ffW-=alpha*self.dffW
		self.ffb-=alpha*self.dffb

	def train(self):
		m=self.X.shape[0]
		batchsize=self.opts['batchsize']
		numbatches = m/batchsize
		print numbatches
		self.rL = []
		for i in range(self.opts['numepochs']):
			print 'the %d -th epoch is running'% (i+1)
			kk=np.random.permutation(m)
			for j in range(numbatches):
				print 'the %d -th batch is running , totally  %d batchs'% ((j+1),numbatches)
				batch_x=self.X[kk[(j)*batchsize:(j+1)*batchsize],:,:].copy()
				batch_y=self.y[:,kk[(j)*batchsize:(j+1)*batchsize]].copy()
				self.cnnff(batch_x)
				self.cnnbp(batch_y)
				self.cnnapplygrads(alpha=self.opts['alpha'])

				if len(self.rL)==0:
					self.rL.append(self.L)
				else:
					p=self.rL[len(self.rL)-1]
					self.rL.append(p*0.99+0.1*self.L)
				print self.L
	def gradcheck(self,test_x,test_y):
		epsilon=0.0001
		er=0.00000001
		n=len(self.layers)
		
		print 'check last bias'
		for i in range(len(self.ffb)):
			temp=self.ffb[i].copy()
			self.ffb[i]+=epsilon
			self.cnnff(test_x)
			self.cnnbp(test_y)
			self.ffb[i]=temp
			L1=self.L
			temp=self.ffb[i].copy()
			self.ffb[i]-=epsilon
			self.cnnff(test_x)
			self.cnnbp(test_y)
			self.ffb[i]=temp
			L2=self.L
			self.cnnff(test_x)
			self.cnnbp(test_y)

			d=(L1-L2)/(2*epsilon)
			e=np.abs(d-self.dffb[i])
			print e
			'''
		print 'check last W'
		print self.ffW.size
		for i in range(self.ffW.shape[0]):
			for j in range(self.ffW.shape[1]):
				temp=self.ffW[i][j].copy()
				self.ffW[i][j]+=epsilon
				self.cnnff(test_x)
				self.cnnbp(test_y)
				self.ffW[i][j]=temp
				L1=self.L
				temp=self.ffW[i][j].copy()
				self.ffW[i][j]-=epsilon
				self.cnnff(test_x)
				self.cnnbp(test_y)
				self.ffW[i][j]=temp
				L2=self.L
				self.cnnff(test_x)
				self.cnnbp(test_y)

				d=(L1-L2)/(2*epsilon)
				e=np.abs(d-self.dffW[i][j])
				print e
	'''
		print 'check each layers'
		self.cnnff(test_x)
		for l in range(1,len(self.layers)):
			print 'layer ',l
			if self.layers[l].types=='c':
				print 'c'
				for j in range(len(self.layers[l].a)):
					for i in range(len(self.layers[l-1].a)):
						for u in range(self.layers[l].k[i][j].shape[0]):
							for v in range(self.layers[l].k[i][j].shape[1]):
								self.layers[l].k[i][j][u][v]+=epsilon
								self.cnnff(test_x)
								self.cnnbp(test_y)
								L1=self.L
								self.layers[l].k[i][j][u][v]-=2*epsilon
								self.cnnff(test_x)
								self.cnnbp(test_y)

								L2=self.L
								self.layers[l].k[i][j][u][v]+=epsilon
								self.cnnff(test_x)
								self.cnnbp(test_y)
								d=(L1-L2)/(2*epsilon)
								e=np.abs(d-self.layers[l].dk[i][j][u][v])
								print e

					self.layers[l].b[j]+=epsilon
					self.cnnff(test_x)
					self.cnnbp(test_y)
					L1=self.L
					self.layers[l].b[j]-=2*epsilon
					self.cnnff(test_x)
					self.cnnbp(test_y)

					L2=self.L
					self.layers[l].b[j]+=epsilon
					self.cnnff(test_x)
					self.cnnbp(test_y)
					d=(L1-L2)/(2*epsilon)
					e=np.abs(d-self.layers[l].db[j])
					print e

	def test(self,test_x,test_y):
		self.cnnff(np.array(test_x))
		p=self.o.argmax(axis=0)
		bad= np.sum(p!=np.array(test_y).argmax(axis=0))
		print p,np.array(test_y).argmax(axis=0)
		print bad
		print np.array(test_y).shape[1]
		er=bad/np.array(test_y).shape[1]
		print er
	def pred(self,test_x):
		self.cnnff(np.array(test_x))
		p=self.o.argmax(axis=0)
		return p