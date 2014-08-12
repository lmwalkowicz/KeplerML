from __future__ import division
import pylab as pl
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN,Ward
from numpy.random import RandomState
rng = RandomState(42)
import itertools
import pyfits
import math
import sys
import heapq
from mpl_toolkits.mplot3d import Axes3D
with open('infiles','r') as f:
	myarray=[line.split() for line in f]
print len(myarray)

"""
test=[]
time=[]
err=[]
teff=[]
for i in range (1000, 2000):
	if i%1==0:
		opens=pyfits.open(myarray[i][0])
		table=opens[1]
		header=table.header
		tabledata=table.data
		test.append(tabledata.field('PDCSAP_FLUX'))
		time.append(tabledata.field('TIME'))
		err.append(tabledata.field('PDCSAP_FLUX_ERR'))
		opens.close()
numpdc=[]
numtime=[]
numerr=[]
print len(test)
for i in range(len(test)):
	numpdc.append([test[i][value] for value in range(len(test[i])) if not ((math.isnan(test[i][value])) or (math.isnan(time[i][value])))])
	numtime.append([time[i][value] for value in range(len(test[i])) if not ((math.isnan(test[i][value])) or (math.isnan(time[i][value])))])
	numerr.append([err[i][value] for value in range(len(test[i])) if not ((math.isnan(test[i][value])) or (math.isnan(time[i][value])))])
yvals=np.zeros(len(numpdc))
"""


"""2d ml plot""" 
#comment

def plot_2D(data, target, target_names):
	colors = itertools.cycle('rgbcmykw')
	target_ids = range(len(target_names))
	pl.figure()
	for i, c, label in zip(target_ids, colors, target_names):
		pl.scatter(data[target == i, 0], data[target == i, 1],c=c, label=label)	
	pl.legend()
	pl.show()

"""3d ml plot"""
def plot_3D(data, target, target_names):
	colors = itertools.cycle('rgbcmykw')
	target_ids = range(len(target_names))
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax = Axes3D(plt.gcf())
	for i, c, label in zip(target_ids, colors, target_names):
		ax.scatter(data[target == i, 0], data[target == i, 1],data[target == i, 2],c=c, label=label)	
	plt.legend()
	plt.show()



#you may then try first8000, etc

with open('first22000','r') as f:
	myarray=[line.split() for line in f]
x=np.array(myarray)
x=x.T

print len(x), len(x[0])

with open('new_file_2.txt','r') as f:
	myarray=[line.split() for line in f]
x=np.array(myarray)
x=x.T

		
print len(x), len(x[0])

#hjkjkhjk


#print len(x), len(x[44000]), len(x[0])
"""
with open('first22000per','r') as f:
	myarrayper=[line.split() for line in f]
xper=np.array(myarrayper)
xper=xper.T
with open('rr.txt','r') as f:
	myarrayrr=[line.split() for line in f]
xrr=np.array(myarrayrr)
xrr=xrr.T
"""

"""print x[0][::]"""
yt=[]
#print len(myarray), len(myarray[44000])

for i in range(len(myarray[0])):
	yt.append([])	
	for j in range(len(myarray)):
		yt[i].append([])
		#print 'i',i,'j',j, x[i][j]
		yt[i][j]=float(x[i][j])
yt=np.array(yt)
y=yt.T
"""
#periodic starts
ytper=[]
for i in range(len(myarrayper[0])):
	ytper.append([])	
	for j in range(len(myarrayper)):
		ytper[i].append([])
		ytper[i][j]=float(xper[i][j])
ytper=np.array(ytper)
yper=ytper.T
#RR Lyr sample
yrr=[]
for i in range(len(myarrayrr[0])):
	yrr.append([])	
	for j in range(len(myarrayrr)):
		yrr[i].append([])
		yrr[i][j]=float(xrr[i][j])
yrr=np.array(yrr)
yrrt=yrr
yrr=yrr.T
#RR 
"""
"""
y=np.vstack((yrr, y))
y=y.T
yt=y
print len(y), len(y[0])
"""
"""
pl.scatter(np.log(abs(yt[::][47]))*abs(yt[::][47])/yt[::][47], yt[::][50])
pl.show()

pl.scatter(np.log(yt[::][13])*yt[::][13]/abs(yt[::][13]), yt[::][26])
pl.show()
"""

"""merge=np.vstack((np.log(abs(yt[::][47]))*abs(yt[::][47])/yt[::][47], yt[::][50]))
merge=merge.T
kmeans = KMeans(n_clusters=4, random_state=rng).fit(merge)
result=np.round(kmeans.cluster_centers_, decimals=2)
result=result.T
results for 6, 12-very good
1, 59- bit confusing
16 with everyone-interesting centre alignments, 1viewing device, 2all needed
34, 37- 	UNIFORM
4, 53
1, 13
remember varrat is now ytper[::][20]/ytper[::][21]
per 67, 2021



"""
"""
#PCA
#merger=np.vstack((np.log(abs(yt[40][::]))*abs(yt[40][::])/yt[40][::], np.log(abs(yt[56][::]))*abs(yt[56][::])/yt[56][::],yt[30][::], yt[25][::], yt[53][::], yt[52][::]))
merger=np.vstack((np.log(abs(yt[56][::]))*abs(yt[56][::])/yt[56][::],yt[30][::], yt[25][::], yt[53][::], yt[52][::], yt[9][::],yt[29][::], yt[43][::],yt[10][::],yt[6][::], yt[19][::],yt[51][::]))
#see 16,10,19 done-52-ok, 

merger=merger.T
pca = PCA(n_components=3, whiten=True).fit(merger)
print pca.components_ 
print pca.explained_variance_ratio_ 
x_pca = pca.transform(merger)

pca = PCA(n_components=12, whiten=True).fit(merger)
print pca.components_ 
print pca.explained_variance_ratio_ 
x_pca_2 = pca.transform(merger)
#KM
yv2=np.zeros(8185)
#merge=np.vstack((yt[::][32], ytper[::][6]))
#print len(y[16][::])
#merge=np.vstack((y[16][::], y[17][::]))
#print np.corrcoef(yt[::][32],ytper[::][6])
#merge=np.vstack((np.log(abs(yt[::][40]))*abs(yt[::][40])/yt[::][40], np.log(abs(yt[::][56]))*abs(yt[::][56])/yt[::][56]))
#merge=merge.T
print i	
kmeans = KMeans(n_clusters=8, random_state=rng).fit(x_pca_2)
result=np.round(kmeans.cluster_centers_, decimals=2)
result=result.T
plot_3D(x_pca,kmeans.labels_, ["c1","c2","c3","c4","c5","c6", "c7","c8"])

print len(x_pca), len(x_pca[0])
"""
#3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax = Axes3D(plt.gcf())

#ax.scatter(merge[0][::], merge[1][::], merge[2][::], zdir='z', s=20, c='b')
#plt.show()

#ax.scatter(np.log(abs(yt[::][40]))*abs(yt[::][40])/yt[::][40], np.log(abs(yt[::][56]))*abs(yt[::][56])/yt[::][56], yt[::][25], c=yt[53][::])
#x_pca=x_pca.T
#ax.scatter(x_pca[::][0],x_pca[::][1] ,x_pca[::][2])
#ax.scatter(np.log(abs(yt[40][::]))*abs(yt[40][::])/yt[40][::], np.log(abs(yt[56][::]))*abs(yt[56][::])/yt[56][::], yt[43][::],c=yt[53][::])
#plt.scatter(yt[34][::], np.log(abs(yt[56][::]))*abs(yt[56][::])/yt[56][::])
#ax.scatter(yt[34][::],yt[37][::],yt[30][::],c=yt[52][::])
ax.scatter(yt[16][::], np.log(abs(yt[56][::]))*abs(yt[56][::])/yt[56][::],yt[30][::],c=yt[52][::])
#ax.scatter(yt[16][::], yt[10][::], yt[19][::]),please try log flatrat,numc-sautocorr
#plt.colorbar
#plt.scatter(np.log(abs(yt[56][::]))*abs(yt[56][::])/yt[56][::],yt[30][::])
#ax.scatter(yt[16][::],np.log(abs(yt[56][::]))*abs(yt[56][::])/yt[56][::],yt[30][::], c=yt[53][::])
plt.show()

#try meanpslope-done, mauto, kurt-done, numspikes, mid20, mid80, roundrat flatrat meanmedrat etc.
# later play with frequency, etc. to determine whats hapenning with the periodogram, through simpleview, change periodic, etc.,create a kmeans 3d viewer

sys.exit()
#DBSCAN
db = DBSCAN(eps=0.05, min_samples=5,metric='euclidean').fit(merger)
core_samples = db.core_sample_indices_
labels = db.labels_
components=db.components_
n_clusters_ = len(set(labels)) 

pl.close('all')
pl.figure(1)
pl.clf()

# Black removed and is used for noise instead.
colors = itertools.cycle('bgrcmybgrcmybgrcmybgrcmy')
for k, col in zip(set(labels), colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
        markersize = 6
    class_members = [index[0] for index in np.argwhere(labels == k)]
    cluster_core_samples = [index for index in core_samples
                            if labels[index] == k]
    for index in class_members:
        x = merger[index]
        if index in core_samples and k != -1:
            markersize = 6
        else:
            markersize = 6
        pl.plot(x_pca[0], x_pca[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=markersize)

pl.title('Estimated number of clusters: %d' % n_clusters_)
pl.show()

sys.exit()







sys.exit()
"""
yv=np.ones(13000)
for i in range(9, 24):
	#merge=np.vstack((np.log(abs(ytper[::][i]))*abs(ytper[::][i])/ytper[::][i], np.log(abs(yv))*abs(yv)/yv))
	#merge=np.vstack((ytper[::][i], np.log(abs(yv))*abs(yv)/yv))
	#print np.corrcoef(yt[::][29], yt[::][56])
	merge=merge.T
	print i	
	kmeans = KMeans(n_clusters=4, random_state=rng).fit(merge)
	result=np.round(kmeans.cluster_centers_, decimals=2)
	result=result.T
	#plt.scatter(result[0],result[1],color='orange')

	plot_2D(merge,kmeans.labels_, ["c1","c2","c3","c4"])


sys.exit()

merge=np.vstack((ytper[::][20], ytper[::][21]))
merge=merge.T
print np.corrcoef(ytper[::][20], ytper[::][21])"""

db = DBSCAN(eps=0.05, min_samples=5,metric='euclidean').fit(merge)
core_samples = db.core_sample_indices_
labels = db.labels_
components=db.components_
n_clusters_ = len(set(labels)) 

pl.close('all')
pl.figure(1)
pl.clf()

# Black removed and is used for noise instead.
colors = itertools.cycle('bgrcmybgrcmybgrcmybgrcmy')
for k, col in zip(set(labels), colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
        markersize = 6
    class_members = [index[0] for index in np.argwhere(labels == k)]
    cluster_core_samples = [index for index in core_samples
                            if labels[index] == k]
    for index in class_members:
        x = merge[index]
        if index in core_samples and k != -1:
            markersize = 6
        else:
            markersize = 6
        pl.plot(x[0], x[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=markersize)

pl.title('Estimated number of clusters: %d' % n_clusters_)
pl.show()

sys.exit()




"""
yvals=np.zeros(1000)
for i in range(60):

	merge=np.vstack(( np.log(abs(yt[::][16]))*abs(yt[::][16])/yt[::][16], yt[::][i] ))
	merge=merge.T
	kmeans = KMeans(n_clusters=4, random_state=rng).fit(merge)
	result=np.round(kmeans.cluster_centers_, decimals=2)
	result=result.T
	print i
	plt.scatter(result[0],result[1],color='orange')

	plot_2D(merge,kmeans.labels_, ["c1","c2","c3","c4"])
	
	
	merge=np.vstack((np.log(abs(yt[::][i]))*abs(yt[::][i])/yt[::][i], yvals))
	merge=merge.T
	kmeans = KMeans(n_clusters=4, random_state=rng).fit(merge)
	result=np.round(kmeans.cluster_centers_, decimals=2)
	result=result.T
	print i
	"""
"""plt.scatter(result[0],result[1],color='orange')"""

"""plot_2D(merge,kmeans.labels_, ["c1","c2","c3","c4"])
corcoef=np.zeros((len(y[0]), len(y[0])))"""

"""print 'one'"""
	

clusters=np.zeros((len(y[0]), len(y[0])))
labels=np.zeros((len(y[0]), len(y[0]), len(y)))

for i in range(len(y[0])):
	for j in range(len(y[0])):
		merge=np.vstack((yt[::][i], yt[::][j]))
		merge=merge.T
		db = DBSCAN(eps=5, min_samples=3,metric='euclidean').fit(merge)
		labels[i][j] = db.labels_
		
		clusters[i][j]= len(set(labels[i][j])) 
		print clusters[i][j]
textfile = file('clusters.txt','wt')
f=open("clusters.txt","w")
for i in range(len(clusters)):
	for j in range(len(clusters[i])):
		f.write(str(clusters[i][j])+' ')
	f.write('\n')
f.close()

textfile = file('labels.txt','wt')
f=open("labels.txt","w")
for i in range(len(labels)):
	for j in range(len(labels[i])):
		f.write(str(labels[i][j])+' ')
	f.write('\n')
f.close()

"""
corcoef=np.zeros((len(y[0]), len(y[0])))
for i in range(len(y[0])):
	for j in range(len(y[0])):
		corcoef[i][j]= np.corrcoef(yt[::][i], yt[::][j])[0][1]

for i in range(len(y[0])):
	corcoef[i][i]=0

textfile = file('corcoef.txt','wt')

print len(xtr)
print len(xtr[0])

f=open("corcoef.txt","w")
for i in range(len(corcoef)):
	for j in range(len(corcoef[i])):
		f.write(str(corcoef[i][j])+' ')
	f.write('\n')
f.close()
"""
"""corcoef.reshape(y[0]*y[0])

corr= np.reshape(corcoef, len(y[0])*len(y[0]))

for j in heapq.nlargest(10, corr):
	print np.argwhere(corcoef==j)
	t= np.argwhere(corcoef==j)
	print j
	plt.scatter(yt[::][t[0][0]], yt[::][t[0][1]])
	plt.show()
"""
with open('clusters.txt','r') as f:
	myarray2=[line.split() for line in f]

for i in range(len(myarray2)):
	for j in range(len(myarray2[0])):
		myarray2[i][j]=float(myarray2[i][j])
"""
with open('labels.txt','r') as f:
	myarray3=[line.split() for line in f]

for i in range(len(myarray3)):
	for j in range(len(myarray3[0])):
		myarray2[i][j]=float(myarray2[i][j])
"""
print len(myarray2)
cluster= np.reshape(myarray2, len(y[0])*len(y[0]))
print max(cluster)

for j in heapq.nlargest(10, cluster):
	"""print np.argwhere(myarray2==j)"""
	t= np.argwhere(myarray2==j)
	print j
	for k in t:
		print k[0], k[1]
		plt.scatter(yt[::][k[0]], yt[::][k[1]])
		plt.show()

sys.exit()	


"""

colors = itertools.cycle('bgrcmybgrcmybgrcmybgrcmy')
for k, col in zip(set(labels), colors):
		if k == -1:
        # Black used for noise.
        		col = 'k'
        		markersize = 6
    		class_members = [index[0] for index in np.argwhere(labels == k)]
    		cluster_core_samples = [index for index in core_samples
                            if labels[index] == k]
    	for index in class_members:
        	x = merge[index]
        	if index in core_samples and k != -1:
            		markersize = 6
        	else:
            		markersize = 6
        	pl.plot(x[0], x[1], 'o', markerfacecolor=col,
                	markeredgecolor='k', markersize=markersize)

	pl.title('Estimated number of clusters: %d' % n_clusters_)
	pl.show()


print len(t), len(t[0])


"""







sys.exit()
corr= np.reshape(corcoef, len(y[0])*len(y[0]))
for j in corr:
	if (True)&(j <=-0.8):
		print np.argwhere(corcoef==j)
		t= np.argwhere(corcoef==j)
		print j
		plt.xlabel(j)
		plt.scatter(yt[::][t[0][0]], yt[::][t[0][1]])
		plt.show()


pl.close('all')
pl.figure(1)
pl.clf()

# Black removed and is used for noise instead.
colors = itertools.cycle('bgrcmybgrcmybgrcmybgrcmy')
for k, col in zip(set(labels), colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
        markersize = 6
    class_members = [index[0] for index in np.argwhere(labels == k)]
    cluster_core_samples = [index for index in core_samples
                            if labels[index] == k]
    for index in class_members:
        x = x_pca[index]
        if index in core_samples and k != -1:
            markersize = 6
        else:
            markersize = 6
        pl.plot(x[0], x[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=markersize)

pl.title('Estimated number of clusters: %d' % n_clusters_)
pl.show()


"""
corr= np.reshape(corcoef, len(y[0])*len(y[0]))
for j in corr:
	if (True)&(j <=-0.8):
		print np.argwhere(corcoef==j)
		t= np.argwhere(corcoef==j)
		print j
		plt.xlabel(j)
		plt.scatter(yt[::][t[0][0]], yt[::][t[0][1]])
		plt.show()


sys.exit()
---------------------------PCA and Clustering Algorithms---------------------------------
nperiodicity=periodicity/max(periodicity)-min(periodicity)

x=np.vstack((periodicity,mautocorrcoef,g_asymm, autocorrcoef,np.log(abs(roundmean))*abs(roundmean)/roundmean,f1,mid20,skewslope,np.log(num_zcross),flatrat))
x2=np.vstack((g_asymm,skewslope))
xnorm=((periodicity/))
xtr=x.T
"""
"""try meanwidth, pasymm, skewslopes, absmeansecder.numnspikes too long(?), logflux, amp10,amp_2, varabsslope-log and otherwise,g_asymm,maxslope, coeffvar,numoutliers,num_zcross,roundmean, f1 try kurt skews, numpm,logamp2, depth*tflatness, pasymm, roundrat and its log, periodogram of maxes"""

"""PCA"""
pca = PCA(n_components=2, whiten=True).fit(merge)
"""print pca.components_ """
print pca.explained_variance_ratio_ 
x_pca = pca.transform(merge)

"""DBSCAN"""
db = DBSCAN(eps=5, min_samples=3,metric='euclidean').fit(merge)
core_samples = db.core_sample_indices_
labels = db.labels_
components=db.components_
n_clusters_ = len(set(labels)) 





"""
print labels

print n_clusters_
"""
"""
print core_samples
print components
"""

pl.close('all')
pl.figure(1)
pl.clf()

# Black removed and is used for noise instead.
colors = itertools.cycle('bgrcmybgrcmybgrcmybgrcmy')
for k, col in zip(set(labels), colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
        markersize = 6
    class_members = [index[0] for index in np.argwhere(labels == k)]
    cluster_core_samples = [index for index in core_samples
                            if labels[index] == k]
    for index in class_members:
        x = x_pca[index]
        if index in core_samples and k != -1:
            markersize = 6
        else:
            markersize = 6
        pl.plot(x[0], x[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=markersize)

pl.title('Estimated number of clusters: %d' % n_clusters_)
pl.show()

"""
for i in range(n_clusters_):
	print '--------------'
"""	


"""KMEANS"""
kmeans = KMeans(n_clusters=4, random_state=rng).fit(merge)
result=np.round(kmeans.cluster_centers_, decimals=2)
"""
result=result.T
print result
"""
plot_2D(merge,kmeans.labels_, ["c1","c2","c3","c4"])

"""WARD"""

ward=Ward(n_clusters=4, connectivity=None, copy=True, n_components=None, compute_full_tree='auto').fit(merge)
"""
print ward.labels_
print ward.n_leaves_
print ward.children_
print ward.n_components_"""

plot_2D(merge,ward.labels_,["c1","c2","c3","c4"])

"""LAST TV

print 'one'
for i in range(len(numpdc)):
	if ward.labels_[i]==0:
		
		plt.scatter(numtime[i],numpdc[i],color='red')
		plt.show()
		plt.xlim(970,975)
		plt.scatter(numtime[i],numpdc[i],color='red')
		plt.show()
print 'two' 
for i in range(len(myarray)):
	if ward.labels_[i]==1:
	
		plt.scatter(numtime[i],numpdc[i],color='red')
		plt.show()
		plt.xlim(970,975)
		plt.scatter(numtime[i],numpdc[i],color='red')
		plt.show()
print 'three' 
for i in range(len(myarray)):	
	if ward.labels_[i]==2:
		print 'three' 
		plt.scatter(numtime[i],numpdc[i],color='red')
		plt.show()
		plt.xlim(970,975)
		plt.scatter(numtime[i],numpdc[i],color='red')
		plt.show()
print 'four' 
for i in range(len(myarray)):
	if ward.labels_[i]==3:
		print 'four'
		plt.scatter(numtime[i],numpdc[i],color='red')
		plt.show()
		plt.xlim(970,975)
		plt.scatter(numtime[i],numpdc[i],color='red')
		plt.show()		 
"""





