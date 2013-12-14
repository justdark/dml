from __future__ import division
import numpy as np
import scipy as sp
from numpy.random import normal,random,uniform
'''
   this code still have some problem,I can only get 0.98 rmse on movielens data
   If you can figure it out,PLEASE!!! tell me . 
'''
class TEMP:
	def __init__(self):
		self.AccVH=None
		self.CountVH=None
		self.AccV=None
		self.temp.CountV=None
		self.temp.AccH=None
class  CF_RMB_C:
	def __init__(self,X,UserNum=943,HiddenNum=30,ItemNum=1682,Rate=5):
		self.X=np.array(X)
		self.HiddenNum=HiddenNum
		self.ItemNum=ItemNum
		self.UserNum=UserNum
		self.Rate=Rate
		self.movie_user={}
		self.user_movie={}
		self.bik=np.zeros((self.ItemNum,self.Rate))
		self.Momentum={}
		self.Momentum['bik']=np.zeros((self.ItemNum,self.Rate))
		self.UMatrix=np.zeros((self.UserNum,self.ItemNum))
		self.V=np.zeros((self.ItemNum,self.Rate))
		for i in range(self.X.shape[0]):
			uid=self.X[i][0]-1
			mid=self.X[i][1]-1
			rat=self.X[i][2]-1
			self.UMatrix[uid][mid]=1
			self.bik[mid][rat]+=1
			self.movie_user.setdefault(mid,{})
			self.user_movie.setdefault(uid,{})
			self.movie_user[mid][uid]=rat
			self.user_movie[uid][mid]=rat
		pass
		self.W=normal(0,0.01,(self.ItemNum,self.Rate,HiddenNum))
		self.Momentum['W']=np.zeros(self.W.shape)
		self.initialize_bik()
		self.bj=np.zeros((HiddenNum,1)).flatten(1)
		self.Momentum['bj']=np.zeros(self.bj.shape)
		self.Dij=np.zeros((self.ItemNum,self.HiddenNum))
		self.Momentum['Dij']=np.zeros((self.ItemNum,self.HiddenNum))
	def initialize_bik(self):
		for i in range(self.ItemNum):
			total=np.sum(self.bik[i])
			if total>0:
				for k in range(self.Rate):
					if self.bik[i][k]==0:
						self.bik[i][k]=-10
					else:
						self.bik[i][k]=np.log(self.bik[i][k]/total)
			
	def test(self,test_X):
		output=[]
		sums=0
		test_X=np.array(test_X)
		#print "the test data size is ",test_X.shape
		for i in range(test_X.shape[0]):
			pre=self.pred(test_X[i][0]-1,test_X[i][1]-1)
			#print test_X[i][2],pre
			output.append(pre)
			#print pre,test_X[i][2]
			sums+=(pre-test_X[i][2])**2
		rmse=np.sqrt(sums/test_X.shape[0])
		print "the rmse on test data is ",rmse
		return output
	def pred(self,uid,mid):
		V=self.clamp_user(uid)
		pj=self.update_hidden(V,uid)
		vp=self.update_visible(pj,uid,mid)
		ans=0
		for i in range(self.Rate):
			ans+=vp[i]*(i+1)
		return ans
	def clamp_user(self,uid):
		V=np.zeros(self.V.shape)
		for i in self.user_movie[uid]:
			V[i][self.user_movie[uid][i]]=1
		return V
	def train(self,para,test_X,cd_steps=3,batch_size=30,numEpoch=100,Err=0.00001):
		for epo in range(numEpoch):
			print 'the ',epo,'-th  epoch is running'
			kk=np.random.permutation(range(self.UserNum))
			bt_count=0
			while bt_count<=self.UserNum:
				btend=min(self.UserNum,bt_count+batch_size)
				users=kk[bt_count:btend]
				temp=TEMP
				temp.AccVH=np.zeros(self.W.shape)
				temp.CountVH=np.zeros(self.W.shape)
				temp.AccV=np.zeros(self.V.shape)
				temp.CountV=np.zeros(self.V.shape)
				temp.AccH=np.zeros(self.bj.shape)
				watched=np.zeros(self.UMatrix[0].shape)
				for user in users:
					watched[self.UMatrix[user]==1]=1
					sv=self.clamp_user(user)
					pj=self.update_hidden(sv,user)
					temp=self.accum_temp(sv,pj,temp,user)
					#AccVH+=pj*
					for step in range(cd_steps):
						sh=self.sample_hidden(pj)
						vp=self.update_visible(sh,user)
						sv=self.sample_visible(vp,user)
						pj=self.update_hidden(sv,user)
					deaccum_temp=self.deaccum_temp(sv,pj,temp,user)
				self.updateall(temp,batch_size,para,watched)	
				#updateall============================================	
				bt_count+=batch_size
			self.test(test_X)

	def accum_temp(self,V,pj,temp,uid):
		for i in self.user_movie[uid]:
			temp.AccVH[i]+=np.dot(V[i].reshape(-1,1),pj.reshape(1,-1))
			temp.CountVH[i]+=1
			temp.AccV[i]+=V[i]
			temp.CountV[i]+=1
		temp.AccH+=pj
		return temp
	def deaccum_temp(self,V,pj,temp,uid):
		for i in self.user_movie[uid]:
			temp.AccVH[i]-=np.dot(V[i].reshape(-1,1),pj.reshape(1,-1))
			temp.AccV[i]-=V[i]
		temp.AccH-=pj
		return temp	
	def updateall(self,temp,batch_size,para,watched):
		delatW=np.zeros(temp.CountVH.shape)
		delatBik=np.zeros(temp.CountV.shape)
	
		delatW[temp.CountVH!=0]=temp.AccVH[temp.CountVH!=0]/temp.CountVH[temp.CountVH!=0]
		delatBik[temp.CountV!=0]=temp.AccV[temp.CountV!=0]/temp.CountV[temp.CountV!=0]
		delataBj=temp.AccH/batch_size

		self.Momentum['W'][temp.CountVH!=0]=self.Momentum['W'][temp.CountVH!=0]*para['Momentum']
		self.Momentum['W'][temp.CountVH!=0]+=para['W']*(delatW[temp.CountVH!=0]-para['weight_cost']*self.W[temp.CountVH!=0])
		self.W[temp.CountVH!=0]+=self.Momentum['W'][temp.CountVH!=0]

		self.Momentum['bik'][temp.CountV!=0]=self.Momentum['bik'][temp.CountV!=0]*para['Momentum']
		self.Momentum['bik'][temp.CountV!=0]+=para['bik']*delatBik[temp.CountV!=0]
		self.bik[temp.CountV!=0]+=self.Momentum['bik'][temp.CountV!=0]

		self.Momentum['bj']=self.Momentum['bj']*para['Momentum']
		self.Momentum['bj']+=para['bj']*delataBj
		self.bj+=self.Momentum['bj']

		for i in range(self.ItemNum):
			if watched[i]==1:
				self.Momentum['Dij'][i]=self.Momentum['Dij'][i]*para['Momentum']
				self.Momentum['Dij'][i]+=para['D']*temp.AccH/batch_size
				self.Dij[i]+=self.Momentum['Dij'][i]
		
	np.seterr(all='raise')
	def update_hidden(self,V,uid):
		r=self.UMatrix[uid]
		hp=None
		for i in self.user_movie[uid]:
			if hp==None:
				hp=np.dot(V[i],self.W[i]).flatten(1)
			else:
				hp+=np.dot(V[i],self.W[i]).flatten(1)
		pj=1/(1+np.exp(-self.bj-hp+np.dot(r,self.Dij).flatten(1)))
		#pj=1/(1+np.exp(-self.bj-hp))
		return pj		
	def sample_hidden(self,pj):
		sh=uniform(size=pj.shape)
		for i in range(sh.shape[0]):
			if sh[i]<pj[i]:
				sh[i]=1.0
			else:
				sh[i]=0.0
		return sh
	def update_visible(self,sh,uid,mid=None):
		if mid==None:
			vp=np.zeros(self.V.shape)
			for i in self.user_movie[uid]:
				
				vp[i]=np.exp(self.bik[i]+np.dot(self.W[i],sh))
				vp[i]=vp[i]/np.sum(vp[i])
			return vp
		vp=np.exp(self.bik[mid]+np.dot(self.W[mid],sh))
		vp=vp/np.sum(vp)
		return vp
	def sample_visible(self,vp,uid):
		sv=np.zeros(self.V.shape)
		for i in self.user_movie[uid]:
			r=uniform()
			k=0
			for k in range(self.Rate):
				r-=vp[i][k]
				if r<=0:break
			sv[i][k]=1
		return sv
