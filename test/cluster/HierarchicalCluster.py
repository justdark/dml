from numpy import array
from scipy.cluster.vq import vq, kmeans, whiten
from dml.CLUSTER import HierarchicalClusterC
import matplotlib.pyplot as plt
features  = array([
[6,11.05],
[7.35,10.6],
[8.35,8.35],
[10.1,7.8],
[10.2,8.7],
[9.25,10],
[8.05,10.8],
[6.6,11.6],
[6.4,9.15],
[8.35,8.25],
[9.5,7.05],
[9.05,8.85],
[8.2,9.45],
[7.15,9.4],
[7.3,7.75],
[8.2,6.7],
[15.45,17.3],
[16.35,16.35],
[17.45,16.5],
[18.05,17.45],
[17.5,18.6],
[16.1,18.9],
[16.75,17.5],
[17.55,17.45],
[15.95,18.35],
[15.2,18],
[15,17.45],
[19.75,8.25],
[20.75,7.9],
[25.65,8.2],
[25.05,10.5],
[22.85,11.2],
[21.6,9.9],
[23.05,8.3],
[24.65,8.8],
[23.55,10.1],
[23.05,9.45],
[23.2,8.35],
[24.2,7.95],
[22.15,7.1],
[21.6,7.8],
[21.7,8.2]]).transpose()
a=HierarchicalClusterC(features,'average')
a.train()
re=a.result(3)

for i in range(len(re)):
	for j in range(re[i].shape[1]):
		if i==0:
			plt.plot(re[i][0][j],re[i][1][j],'or')
		elif i==1:
			plt.plot(re[i][0][j],re[i][1][j],'ob')
		elif i==2:
			plt.plot(re[i][0][j],re[i][1][j],'oy')
plt.show()

#print a.result()
#print a.bfWhiteCen()