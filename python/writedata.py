# reads in both both .sav files: the long-cadence data and the variable supplement
# removes all objects that lack data like effective temperatures
# writes the dataset.npy and logdata.npy files
# also building the training and test sets and runs a Gaussian Naive Bayes classifier

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
#from __future__ import print_function

s = readsav('../data/grndsts9_vars.sav')
s2 = readsav('../data/ts_stats.sav')

#print int(s2.kepid[0]), s2.kepmag[0], s2.teff[0], s2.logg[0], s2.radius[0], s2.range[0],     s2.rmsmdv[0], s2.mdv3[0], s2.mdv6[0], s2.mdv12[0], s2.mdv24[0], s2.mdv48[0], s2.mdv4d[0], s2.mdv8d[0],     s2.npk[0], s2.maxpkht[0], s2.maxper[0], s2.maxflx[0], s2.sndht[0], s2.sndper[0], s2.sndflx[0], s2.np3d[0], s2.np9d[0], s2.nsigpks[0],     s2.rms4[0], s2.cdpp3[0], s2.cdpp6[0], s2.cdpp12[0], s2.pdcvar[0], s2.crowd[0]
print len(s.teff)
n1 = len([x for x in s.teff if x>0])
print n1
for i in range(0,n1):
    if s.teff[i]>0 and s.cdpp3[i]==0: n1 = n1-1
print n1
n1 = 151797

n2 = n1 + len([x for x in s2.teff if x>0])
print n2
for i in range(n1,n2):
    if s2.teff[i-n1]>0 and s2.cdpp3[i-n1]==0: n2 = n2-1
print n2
dataset = np.empty([n2,30])

i=0
j=0
while i<n1:
    if s.teff[j]>0 and s.cdpp3[j]!=0:
        dataset[i,0] = s.kepid[j]
        dataset[i,1] = s.kepmag[j]
        dataset[i,2] = s.teff[j]
        dataset[i,3] = s.logg[j]
        dataset[i,4] = s.radius[j]
        dataset[i,5] = s.range[j]
        dataset[i,6] = s.rmsmdv[j]
        dataset[i,7] = s.mdv3[j]
        dataset[i,8] = s.mdv6[j]
        dataset[i,9] = s.mdv12[j]
        dataset[i,10] = s.mdv24[j]
        dataset[i,11] = s.mdv48[j]
        dataset[i,12] = s.mdv4d[j]
        dataset[i,13] = s.mdv8d[j]
        dataset[i,14] = s.npk[j]
        dataset[i,15] = s.maxpkht[j]
        dataset[i,16] = s.maxper[j]
        dataset[i,17] = s.maxflx[j]
        dataset[i,18] = s.sndht[j]
        dataset[i,19] = s.sndper[j]
        dataset[i,20] = s.sndflx[j]
        dataset[i,21] = s.np3d[j]
        dataset[i,22] = s.np9d[j]
        dataset[i,23] = s.nsigpks[j]
        dataset[i,24] = s.rms4[j]
        dataset[i,25] = s.cdpp3[j]
        dataset[i,26] = s.cdpp6[j]
        dataset[i,27] = s.cdpp12[j]
        dataset[i,28] = s.pdcvar[j]
        dataset[i,29] = s.crowd[j]
        i=i+1
    j=j+1
    if i>152780:
        print i,j
        print s.kepid[j]
        print dataset[i]

print dataset[895]

i=n1
j=0
while i<n2:
    if s2.teff[j]>0 and s2.cdpp3[j]!=0:
        dataset[i,0] = s2.kepid[j]
        dataset[i,1] = s2.kepmag[j]
        dataset[i,2] = s2.teff[j]
        dataset[i,3] = s2.logg[j]
        dataset[i,4] = s2.radius[j]
        dataset[i,5] = s2.range[j]
        dataset[i,6] = s2.rmsmdv[j]
        dataset[i,7] = s2.mdv3[j]
        dataset[i,8] = s2.mdv6[j]
        dataset[i,9] = s2.mdv12[j]
        dataset[i,10] = s2.mdv24[j]
        dataset[i,11] = s2.mdv48[j]
        dataset[i,12] = s2.mdv4d[j]
        dataset[i,13] = s2.mdv8d[j]
        dataset[i,14] = s2.npk[j]
        dataset[i,15] = s2.maxpkht[j]
        dataset[i,16] = s2.maxper[j]
        dataset[i,17] = s2.maxflx[j]
        dataset[i,18] = s2.sndht[j]
        dataset[i,19] = s2.sndper[j]
        dataset[i,20] = s2.sndflx[j]
        dataset[i,21] = s2.np3d[j]
        dataset[i,22] = s2.np9d[j]
        dataset[i,23] = s2.nsigpks[j]
        dataset[i,24] = s2.rms4[j]
        dataset[i,25] = s2.cdpp3[j]
        dataset[i,26] = s2.cdpp6[j]
        dataset[i,27] = s2.cdpp12[j]
        dataset[i,28] = s2.pdcvar[j]
        dataset[i,29] = s2.crowd[j]
        i=i+1
    j=j+1

q=8    
logdata = np.log10(dataset)
testdata = np.empty([n2,q])
plotdata = np.empty([n2,q])

plotdata[:,0] = dataset[:,2]  #teff
plotdata[:,1] = dataset[:,3]  #logg
plotdata[:,2] = logdata[:,15] #maxpkht
plotdata[:,3] = logdata[:,18] #sndht
plotdata[:,4] = logdata[:,16] #maxper
plotdata[:,5] = logdata[:,19] #sndper
plotdata[:,6] = logdata[:,5]  #range
plotdata[:,7] = logdata[:,24] #rms4


np.save('../data/dataset',dataset)
np.save('../data/logdata',logdata)
#s.range = s.range/s.totf

j=0
for i in [2,3,15,18,16,19,5,24]:
    mean = np.mean(logdata[:,i])
    std = np.mean(logdata[:,i])
    testdata[:,j] = (logdata[:,i]-mean)/std
    j=j+1
    
teff = dataset[:,2]

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

num = 100
sunlike = [0]*num
kdwarf = [0]*num
giant = [0]*num
other = [0]*num
stest = [0]*num
ktest = [0]*num
gtest = [0]*num
otest = [0]*num

sun=0
kdw=0
gnt=0
oth=0
j=0
while 1:
    if 3500<=teff[j]<=5100 and (s.logg[j]>=4.2 or s.logg[j]>=(2.2+0.0005*teff[j])):
        if kdw<100: kdwarf[kdw] = dataset[j,0]
        elif kdw<200: ktest[kdw-100] = dataset[j,0]
        kdw = kdw+1
    elif 5600<=teff[j]<=5900:
        if sun<100: sunlike[sun] = dataset[j,0]
        elif sun<200: stest[sun-100] = dataset[j,0]
        sun = sun+1
    elif 0<s.logg[j]<=(6.0-0.0004*teff[j]) and s.logg[j]<=4.0 and teff[j]>0:
        if gnt<100: giant[gnt] = dataset[j,0]
        elif gnt<200: gtest[gnt-100] = dataset[j,0]
        gnt = gnt+1
    elif teff[j]>0:
        if oth<100: other[oth] = dataset[j,0]
        elif oth<200: otest[oth-100] = dataset[j,0]
        oth = oth+1
    if kdw>=200 and sun>=200 and gnt>=200 and oth>=200: break
    j = j+1

#p = len(rrlyrae + instrip + detached + semidet + overcontact + ellipsoid + uncertain + sunlike + kdwarf + giant + other)
#kidlist = [other, sunlike, kdwarf, giant, rrlyrae, instrip, detached, semidet, overcontact, ellipsoid, uncertain]
#testlist = [otest, stest, ktest, gtest, rtest, itest, dtest, sdtest, octest, eltest, utest]
p = len(rrlyrae + instrip + detached + semidet + overcontact + ellipsoid + uncertain)
kidlist = [rrlyrae, instrip, detached, semidet, overcontact, ellipsoid, uncertain]
testlist = [rtest, itest, dtest, sdtest, octest, eltest, utest]
col=np.empty([p],dtype='S10')

training = np.empty([p,q])
pd = np.empty([p,q])
h = 0
for i in kidlist:
    for j in i:
        #print h, j
        temp = testdata[np.where(dataset[:,0]==j)[0][0]]
        temp2 = plotdata[np.where(dataset[:,0]==j)[0][0]]
        for k in range(0,q):
            training[h,k] = temp[k]
        for k in range(0,q):
            pd[h,k] = temp2[k]
        if i==instrip: col[h] = 'blue'
        elif i==detached: col[h] = 'cyan'
        elif i==sunlike: col[h] = 'white'
        elif i==kdwarf: col[h] = 'white'
        elif i==giant: col[h] = 'white'
        elif i==other: col[h] = 'white'
        elif i==rrlyrae: col[h] = 'magenta'
        elif i==semidet: col[h] = 'green'
        elif i==overcontact: col[h] = 'yellow'
        elif i==ellipsoid: col[h] = 'orange'
        elif i==uncertain: col[h] = 'red'
        h = h+1

h=0
sampletest = np.empty([p,q])
pdtest = np.empty([p,q])
coltest=np.empty([p],dtype='S10')
for i in testlist:
    for j in i:
        #print h, j
        temp = testdata[np.where(dataset[:,0]==j)[0][0]]
        temp2 = plotdata[np.where(dataset[:,0]==j)[0][0]]
        for k in range(0,q):
            sampletest[h,k] = temp[k]
        for k in range(0,q):
            pdtest[h,k] = temp2[k]
        if i==itest: coltest[h] = 'blue'
        elif i==dtest: coltest[h] = 'cyan'
        elif i==stest: coltest[h] = 'yellow'
        elif i==ktest: coltest[h] = 'orange'
        elif i==gtest: coltest[h] = 'red'
        elif i==otest: coltest[h] = 'white'
        elif i==rtest: coltest[h] = 'magenta'
        elif i==sdtest: coltest[h] = 'green'
        elif i==octest: coltest[h] = 'yellow'
        elif i==eltest: coltest[h] = 'orange'
        elif i==utest: coltest[h] = 'red'
        h = h+1

#q = len([x for x in testdata if x[0]>0])
'''
q=5000
i=0
j=0
sampletest = np.empty([q,n])
pdtest = np.empty([q,m])
#col=np.empty([q],dtype='S10')
while i<len(sampletest):
    if testdata[j,0] > -10 and j>10000:
        sampletest[i] = testdata[j]
        pdtest[i] = plotdata[j]
        i=i+1
    j=j+1
'''
sample = training
'''
sample2 = np.empty([len(sample),q])
for i in range(0,q):
    temp = sample[:,i].argsort()
    sample2[:,i] = np.arange(len(sample[:,i]))[temp.argsort()]

sample2 = sample2/len(sample)
'''
print np.shape(sample)

fig = pl.figure(1)
pl.clf()
'''
colors = itertools.cycle('bgrcmybgrcmybgrcmybgrcmy')
for k, col in zip(set(labels), colors):
    for index in class_members:
        x = testdata[:,index]
        pl.plot(x[0], x[1], 'o', markerfacecolor=col,
                markeredgecolor=col, markersize=1)
        pl.plot(0,0)
pl.show()

i=1
while i<=1:
    clf = KMeans(n_clusters=i, max_iter=1, random_state=0)
    clf.fit(sample)
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
'''
@pickle_results('forest.pkl')
def compute_forest(depth):
    rms_test = np.zeros(len(depth))
    rms_train = np.zeros(len(depth))
    i_best = 0

    clf = RandomForestClassifier(n_estimators=1, max_depth=5,
                                 min_samples_split=1, random_state=0)
    stuff = clf.fit(rms_train)
    stuff = clf.apply(rms_test)
    print stuff

compute_forest([0,1,2,3,4])

    for i, d in enumerate(depth):
        clf = RandomForestClassifier(n_estimators=10, max_depth=d, 
                                     min_samples_split=1, random_state=0)
        stuff = cross_val_score(clf, rms_test, rms_train)
        print stuff
'''
'''
def onpick(event):
    ind=event.ind
    s1 = ax1.scatter(X[ind,0],X[ind,1],c=col[ind],lw=1,s=36)
    s2 = ax2.scatter(pd[ind,0],pd[ind,1],c=col[ind],lw=1,s=36)
    s3 = ax3.scatter(pd[ind,4],pd[ind,5],c=col[ind],lw=1,s=36)
    s4 = ax4.scatter(pd[ind,6],pd[ind,7],c=col[ind],lw=1,s=36)
    pl.ion()
    pl.draw()
    s1.set_visible(False)
    s2.set_visible(False)
    s3.set_visible(False)
    s4.set_visible(False)
'''

ax1 = pl.subplot(221,axisbg='black')
ax1.scatter(pdtest[:,0],pdtest[:,1],c=y_pred,lw=0,s=7,picker=True)
#ax1.scatter(sample[:,0],sample[:,1],c=col,lw=0,s=7,picker=True)                
#fig.canvas.mpl_connect('pick_event',onpick)
#pl.xlim(2000,12000)
#pl.ylim(0,6)
pl.xlabel('T_eff')
pl.ylabel('log(g)')

ax2 = pl.subplot(222,axisbg='black')
ax2.scatter(pdtest[:,2],pdtest[:,3],c=y_pred,lw=0,s=7,picker=True)
#ax2.scatter(sample[:,2],sample[:,3],c=col,lw=0,s=7,picker=True)                
#fig.canvas.mpl_connect('pick_event',onpick)
#pl.xlim(-1,4)
#pl.ylim(-1,3.5)
pl.xlabel('log-first peak height')
pl.ylabel('log-second peak height')

ax3 = pl.subplot(223,axisbg='black')
ax3.scatter(pdtest[:,4],pdtest[:,5],c=y_pred,lw=0,s=7,picker=True)
#ax3.scatter(sample[:,4],sample[:,5],c=col,lw=0,s=7,picker=True)                
#fig.canvas.mpl_connect('pick_event',onpick)
#pl.xlim(-1.5,2.5)
#pl.ylim(-1.5,2.5)
pl.xlabel('log-first peak period (d)')
pl.ylabel('log-second peak period (d)')

ax4 = pl.subplot(224,axisbg='black')
ax4.scatter(pdtest[:,6],pdtest[:,7],c=y_pred,lw=0,s=7,picker=True)
#ax4.scatter(sample[:,6],sample[:,7],c=col,lw=0,s=7,picker=True)                
#fig.canvas.mpl_connect('pick_event',onpick)
#pl.xlim(-4,2)
#pl.ylim(-3,3)
pl.xlabel('log-amplitude')
pl.ylabel('RMS deviation')

'''
ax1 = pl.subplot(221, axisbg='black')
ax1.scatter(X[:,0],X[:,1],c=col,s=7,lw=0,picker=True)
fig.canvas.mpl_connect('pick_event',onpick)
pl.xlabel('component 1')
pl.ylabel('component 2')

ax2 = pl.subplot(223, sharex=ax1, axisbg='black')
ax2.scatter(X[:,0],X[:,2],c=col,s=7,lw=0)
pl.xlabel('component 1')
pl.ylabel('component 3')

ax3 = pl.subplot(224, sharey=ax2, axisbg='black')
ax3.scatter(X[:,1],X[:,2],c=col,s=7,lw=0)
pl.xlabel('component 2')
'''
pl.show()
