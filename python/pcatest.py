# computes a PCA of the training set or data set and prints the components
# for the purpose of assessing the most significant parameters

import numpy as np
import pyfits as pf
import itertools
from scipy.io.idl import readsav
from scipy.spatial import distance
import pylab as pl
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GMM
from astroML.datasets import fetch_sdss_specgals
from astroML.decorators import pickle_results
from sklearn import naive_bayes
from sklearn.decomposition import PCA
#from __future__ import print_function

#kepid, kepmag, teff, logg, radius, range, 0-5
#rmsmdv, mdv3, mdv6, mdv12, mdv24, mdv48, mdv4d, mdv8d, 6-13
#npk, maxpkht, maxper, maxflx, sndht, sndper, sndflx, np3d, np9d, nsigpks, 14-23
#rms4, cdpp3, cdpp6, cdpp12, pdcvar, crowd, 24-29

dataset = np.load('../data/dataset.npy')
logdata = np.load('../data/logdata.npy')

n2=len(dataset)
print n2
q=21
testdata = np.empty([n2,q])
plotdata = np.empty([n2,q])
'''
plotdata[:,0] = dataset[:,2]  #teff
plotdata[:,1] = dataset[:,3]  #logg
plotdata[:,2] = logdata[:,15] #maxpkht
plotdata[:,3] = logdata[:,18] #sndht
plotdata[:,4] = logdata[:,16] #maxper
plotdata[:,5] = logdata[:,19] #sndper
plotdata[:,6] = logdata[:,5]  #range
plotdata[:,7] = logdata[:,24] #rms4
'''
plotdata[:,0] = logdata[:,7]  #teff
plotdata[:,1] = logdata[:,25]  #logg
plotdata[:,2] = logdata[:,17] #maxpkht
plotdata[:,3] = logdata[:,25] #sndht
plotdata[:,4] = logdata[:,20] #maxper
plotdata[:,5] = logdata[:,25] #sndper
plotdata[:,6] = logdata[:,28]  #range
plotdata[:,7] = logdata[:,25] #rms4

#np.save('dataset',dataset)
#np.save('logdata',logdata)
#s.range = s.range/s.totf

j=0
for i in [2,3,5,7,8,9,10,11,12,13,15,16,17,18,19,20,24,25,26,27,28]:
    mean = np.mean(logdata[:,i])
    std = np.mean(logdata[:,i])
    testdata[:,j] = (logdata[:,i]-mean)/std
    j=j+1
'''
j=0
for i in [14,21,22,23,23,25,26,27]:
    mean = np.mean(dataset[:,i])
    std = np.mean(dataset[:,i])
    testdata[:,j] = (dataset[:,i]-mean)/std
    j=j+1
'''
params = np.empty([len(logdata),21])
j=0
for i in [2,3,5,7,8,9,10,11,12,13,15,16,17,18,19,20,24,25,26,27,28]:
    params[:,j] = logdata[:,i]
    j=j+1
for i in range(0,len(params)):
    for j in range(0,len(params[i])):
        if np.isnan(params[i,j]) or np.isinf(params[i,j]): print i, j, params[i,j]

teff = dataset[:,2]
logg = dataset[:,3]

rrlyrae = [5520878, 3733346, 5299596, 6070714, 6100702,   6763132, 6936115, 7176080]
rtest = [7742534, 7988343, 8344381, 9508655, 9591503,   9947026, 10789273, 11802860]

instrip = [2571868, 2987660, 3629496, 5356349, 5437206,   6668729, 7304385, 7974841, 8018827, 8324268]
itest = [8351193, 8489712, 8915335, 9291618, 9351622,   10537907, 10974032, 11572666, 11874676, 12153021]

detached = [1026032, 1026957, 1433962, 1571511, 1725193,   1996679, 2010607, 2162635, 2162994, 2305372,   2305543, 2306740]
dtest = [2308957, 2309587, 2309719, 2437452, 2438070,   2440757, 2442084, 2445134, 2447893, 2556127,   2557430, 2576692]

semidet = [4947528, 4949770, 5077994, 5120793, 5211385,   5215999, 5218441, 5374999, 5471619, 5774375,   5785586, 5792093, 5809827]
sdtest = [5823121, 6283224, 6302051, 6353203, 6432059,   6606653, 6669809, 6692340, 6836140, 6852488,   6865626, 6962901, 7031714]

overcontact = [7821450, 7830460, 7835348, 7839027, 7871200,   7877062, 7878402, 7879404, 7881722, 7889628,   7950962, 7973882, 7977261]
octest = [8004839, 8035743, 8039225, 8053107, 8108785,   8111387, 8122124, 8143757, 8177958, 8190491,   8190613, 8192840, 8241252]

ellipsoid = [9848190, 9898401, 9909497, 9948201, 10028352,   10030943, 10032392, 10123627, 10135584, 10148799,   10155563, 10285770, 10288502, 10291683, 10351735,   10417135]
eltest = [10481912, 10600319, 10619506, 10855535, 11135978,   11336707, 11572643, 11714337, 11722816, 11751847,   11825204, 11875706, 12055421, 12059158, 12121738,   12166770]

uncertain = [9237533, 9347868, 9347955, 9456920, 9469350,   9480516, 9532591, 9596355, 9655187, 9713664,   9716456, 9724080]
utest = [9724220, 9832227, 9835416, 9874575, 9964422,   10086746, 10264744, 10350225, 10388897, 10556068,   10684673, 10799558]

#candidate = [1027438, 1161345, 1431122, 1432214, 1432789,   1717722, 1718189, 1718958, 1721157, 1724719,   1725016, 1849702, 1865042, 1871056, 1872821,   1995519, 1996180, 2141783, 2142522, 2161536,   2162635, 2164169, 2165002, 2302548, 2303903,   2304320, 2306756, 2307199, 2307415, 2309719,   2438264, 2438513, 2439243, 2441495, 2442448,   2444412, 2449431]

#falsepos = [892772, 1026957, 1433962, 1571511, 1722276,   1996679, 2157247, 2166206, 2309585, 2438070,   2440757, 2441151, 2441728, 2445129, 2445154,   2446113, 2452450]

quiet = [0]*100
qtest = [0]*100

qui=0
j=0
while 1:
    if (logg[j]>=(6.0-0.0004*teff[j]) or logg[j]>=4.0) and logdata[j,5]<=1 and logdata[j,15]<=1:
        if qui<100: quiet[qui] = dataset[j,0]
        elif qui<200: qtest[qui-100] = dataset[j,0]
        qui = qui+1
    if qui>=200: break
    j = j+1

#p = len(rrlyrae + instrip + detached + semidet + overcontact + ellipsoid + uncertain + quiet)
#kidlist = [quiet, rrlyrae, instrip, detached, semidet, overcontact, ellipsoid, uncertain]
#testlist = [qtest, rtest, itest, dtest, sdtest, octest, eltest, utest]
p = len(rrlyrae + instrip + detached + semidet + overcontact + ellipsoid + uncertain)
kidlist = [rrlyrae, instrip, detached, semidet, overcontact, ellipsoid, uncertain]
testlist = [rtest, itest, dtest, sdtest, octest, eltest, utest]
col=np.empty([p],dtype='S10')

params3 = np.empty([len(params),2])
params3[:,0] = params[:,0]
params3[:,1] = params[:,1]
#params3[:,2] = params[:,11]
#params4[:,3] = params[:,14]
training = np.empty([p,q])
pd = np.empty([p,q])
h = 0
for i in kidlist:
    for j in i:                                                             
        temp = params[np.where(dataset[:,0]==j)[0][0]]
        temp2 = plotdata[np.where(dataset[:,0]==j)[0][0]]
        for k in range(0,q):
            training[h,k] = temp[k]
        for k in range(0,q):
            pd[h,k] = temp2[k]
        h = h+1

map = PCA(n_components=21)

#X = map.fit_transform(params)
#tempcomponents = map.components_
#tempratio = map.explained_variance_ratio_
#variance = np.empty([19,1])
#print params3

j=0
params2 = np.empty([len(training),12])
for i in range(0,21):
    if i<=3 or 10<=i<=16 or i==20:
        params2[:,j] = training[:,i]
        j=j+1
    
X = map.fit_transform(params2)

print map.explained_variance_ratio_
print map.explained_variance_ratio_.sum()

fig = pl.figure(1)
pl.clf()

#teff, logg, range, mdv3, mdv6, mdv12, mdv24, mdv48, mdv4d, mdv8d
#maxpkht, maxper, maxflx, sndht, sndper, sndflx
#rms4, cdpp3, cdpp6, cdpp12, pdcvar

l = pl.plot(np.zeros([12]),c='black')
l0 = pl.plot(map.components_[8],c='blue')
l1 = pl.plot(map.components_[9],c='green')
l2 = pl.plot(map.components_[10],c='red')
l3 = pl.plot(map.components_[11],c='orange')
pl.show()
'''
X_train = training
y_train = col
X_test = sampletest
y_test = coltest

gnb = naive_bayes.GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
n=0
for i in range(0,len(y_pred)):
    if y_pred[i] != y_test[i]:
        #print i, y_pred[i], y_test[i]
        n = n+1
print n, 1.0 - n*1.0/len(y_test)

samplecol = gnb.predict(sample)

ax1 = pl.subplot(221,axisbg='black')
ax1.scatter(pd[:,0],pd[:,1],c=col,lw=0,s=7,picker=True)
#ax1.scatter(sample[:,0],sample[:,1],c=col,lw=0,s=7,picker=True)                
#fig.canvas.mpl_connect('pick_event',onpick)
#pl.xlim(2000,12000)
#pl.ylim(0,6)
pl.xlabel('T_eff')
pl.ylabel('log(g)')

ax2 = pl.subplot(222,axisbg='black')
ax2.scatter(pd[:,2],pd[:,3],c=col,lw=0,s=7,picker=True)
#ax2.scatter(sample[:,2],sample[:,3],c=col,lw=0,s=7,picker=True)                
#fig.canvas.mpl_connect('pick_event',onpick)
#pl.xlim(-1,4)
#pl.ylim(-1,3.5)
pl.xlabel('log-first peak height')
pl.ylabel('log-second peak height')

ax3 = pl.subplot(223,axisbg='black')
ax3.scatter(pd[:,4],pd[:,5],c=col,lw=0,s=7,picker=True)
#ax3.scatter(sample[:,4],sample[:,5],c=col,lw=0,s=7,picker=True)                
#fig.canvas.mpl_connect('pick_event',onpick)
#pl.xlim(-1.5,2.5)
#pl.ylim(-1.5,2.5)
pl.xlabel('log-first peak period (d)')
pl.ylabel('log-second peak period (d)')

ax4 = pl.subplot(224,axisbg='black')
ax4.scatter(pd[:,6],pd[:,7],c=col,lw=0,s=7,picker=True)
#ax4.scatter(sample[:,6],sample[:,7],c=col,lw=0,s=7,picker=True)                
#fig.canvas.mpl_connect('pick_event',onpick)
#pl.xlim(-4,2)
#pl.ylim(-3,3)
pl.xlabel('log-amplitude')
pl.ylabel('RMS deviation')

pl.show()
'''
