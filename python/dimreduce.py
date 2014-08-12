# computes and graphs a PCA of a sample of light curves
# for the purposes of dimensionality reduction

import numpy as np
import pyfits as pf
import itertools
from scipy.io.idl import readsav
from scipy.spatial import distance
import pylab as pl
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GMM
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from astroML.datasets import fetch_sdss_specgals
from astroML.decorators import pickle_results
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.manifold import LocallyLinearEmbedding, Isomap
#from __future__ import print_function

s = readsav('../data/grndsts9_vars.sav')

mp = np.log10(s.maxper)
sp = np.log10(s.sndper)
mh = np.log10(s.maxpkht)
sh = np.log10(s.sndht)
s.maxper = np.log10(s.maxper)
s.sndper = np.log10(s.sndper)
s.maxpkht = np.log10(s.maxpkht)
s.sndht = np.log10(s.sndht)
s.range = s.range/s.totf
#r4 = s.rms4/s.totf
r4 = s.rms4
teff = s.teff*1
rad = s.radius*1
amp = s.range*1

for i in range(0,len(amp)):
    if amp[i]>0 and rad[i]>0:
        amp[i] = np.log10(amp[i])
        rad[i] = np.log10(rad[i])

i=0
while i < len(s.teff): 
    if s.teff[i] > 0:
        s.teff[i] = np.log10(s.teff[i])
        s.range[i] = np.log10(s.range[i])
    if s.rms4[i] > 0:
        s.rms4[i] = np.log10(s.rms4[i])
    i=i+1

tm = np.mean([x for x in s.teff if x!=0])
ts = np.std([x for x in s.teff if x!=0])
ampm = np.mean([x for x in s.range if x!=0])
amps = np.std([x for x in s.range if x!=0])
lm = np.mean([x for x in s.logg if x!=0])
ls = np.std([x for x in s.logg if x!=0])
s.maxpkht = (s.maxpkht-np.mean(s.maxpkht))/np.std(s.maxpkht)
s.maxper = (s.maxper-np.mean(s.maxper))/np.std(s.maxper)
s.sndht = (s.sndht-np.mean(s.sndht))/np.std(s.sndht)
s.sndper = (s.sndper-np.mean(s.sndper))/np.std(s.sndper)
teff2 = (s.teff-tm)/ts
logg2 = (s.logg-lm)/ls
amp2 = (s.range-ampm)/amps
rms4 = (s.rms4-np.mean(s.rms4))/np.std(s.rms4)

plotdata = [teff, s.logg, mh, sh, mp, sp, amp, r4]
m=len(plotdata)
plotdata = np.transpose(plotdata)
testdata = [teff2, logg2, s.maxpkht, s.sndht, s.maxper, s.sndper, amp2, rms4]
n=len(testdata)
testdata = np.transpose(testdata)

# 4 pulsating, 100 eclipsing, 100 sun-like, 100 k-dwarfs, 100 giants
instrip = [2571868, 5356349, 5437206, 8489712]

eclipse = [8912468, 10855535, 9612468, 9898401, 7375612,   12350008, 6287172, 11825204, 4921906, 8288741,   12055255, 8108785, 1572353, 10288502, 9238207,   8555795, 6144827, 10030943, 10965091, 2715417,   11413213, 9077796, 6050116, 6350020, 7198474,   7546791, 3972629, 9345163, 8816790, 11494583,   4738426, 8122124, 9032671, 5960283, 9412114,   9004380, 4857282, 12602985, 11336707, 12104285,   11769739, 4138301, 9478836, 12508348, 6871716,   3839964, 12598713, 9700154, 9662581, 2856960,   7339345, 9288175, 9392331, 9239684, 11284547,   8045121, 10557008, 5785551, 9388303, 2437038,   7269843, 6072578, 9026766, 10802917, 3832382,   7697065, 9760531, 5956588, 2448320, 8739802,   4385109, 4563150, 5104097, 7680593, 3853259,   11566174, 3354616, 10074939, 2570289, 11246163,   4476900, 9508052, 9357030, 5786545, 9527167,   9179806, 7272739, 10600319, 5218385, 7269797,   5891963, 7199353, 10350225, 8242493, 4241946,   11405559, 8677949, 11036301, 7367833, 5459373]

sunlike = [9402544, 5640244, 11304194, 5369454, 2020086,   5878913, 10154417, 8866956, 10252521, 3648333,   10683445, 11034017, 11447613, 3118796, 8378779,   5097045, 9820618, 5521442, 4751659, 11124274,   5352130, 8572746, 8155776, 4557072, 3218513,   3655052, 11197005, 892977, 10149905, 11304779,   11241419, 5007471, 4252873, 7739563, 5375035,   8029017, 3229706, 4349736, 5734940, 8754254,   3663494, 10219365, 6590409, 7354587, 5695004,   8085053, 12356937, 11876270, 11407841, 4848497,   5611665, 10903132, 8415200, 9767803, 3348483,   8804764, 10677186, 7517664, 3218489, 12460038,   10027733, 11125962, 7948588, 8759831, 7265433,   10790476, 10924414, 11920065, 3865180, 7875139,   3341249, 10546672, 9007306, 11351200, 7756956,   5301750, 8957663, 9908470, 10815701, 7619004,   7950923, 6974841, 6431250, 10357182, 5872139,   3102541, 8197696, 5288099, 9702895, 9205138,   5031583, 7937360, 7848826, 8266042, 7661097,   11518142, 10854310, 9697090, 9592370, 12553011]

kdwarf = [2711088, 5481021, 6351151, 10908779, 7941412,   10556201, 10264852, 9512360, 9162025, 10963092,   4939907, 11020073, 7834941, 9722513, 9329586,   5094903, 4651312, 12102598, 11128748, 9569397,   4265713, 11401160, 11133844, 6151053, 7456580,   11145556, 12118164, 10273194, 7593903, 5784777,   8414716, 4730961, 4279048, 6945737, 3561525,   8980730, 3431530, 8331188, 8555903, 10256970,   10002597, 12166343, 11619032, 7362434, 8482464,   8226949, 12207283, 11751991, 10590402, 9112164,   11297936, 3234193, 2858337, 10748393, 10866057,   9307131, 11468870, 2304757, 4385594, 2574216,   10092136, 7591637, 5218109, 7938499, 5780269,   7691983, 1849846, 10284971, 5696108, 7538946,   3239770, 10487796, 10601075, 1571340, 10073680,   4937606, 10195818, 4665816, 12012281, 8604666,   11820922, 6862174, 9183248, 4569334, 8716714,   9031452, 8631563, 3728432, 11018253, 12401216,   8832146, 10452526, 4157460, 7907093, 5461988,   10983796, 3627577, 9674770, 3647136, 7987781]

giant = [4914151, 7971070, 11134982, 1431316, 6801204,   9726997, 8773948, 6358970, 8452840, 10685068,   8111555, 5802193, 5446242, 6877290, 11152894,   5396936, 10801792, 11605089, 4736439, 8420916,   7696976, 6525397, 5523030, 11759700, 6524878,   8806387, 2581554, 11341730, 8804145, 2995050,   7120976, 6762447, 7630335, 10014893, 4656246,   11911699, 8077790, 8256572, 7537808, 9368745,   6519747, 9782964, 8613617, 2574191, 11299524,   8936347, 10318539, 10908685, 2014377, 3962731,   2852181, 11392017, 11403918, 7690810, 4383902,   5978547, 7265993, 9175366, 5450141, 4243819,   8824264, 10743331, 10162765, 7377422, 9182085,   9713434, 8782196, 4038624, 12602978, 8058062,   9699882, 9490342, 5018570, 3356237, 11496193,   8086682, 7667124, 10881149, 8818614, 4068867,   10284878, 10068490, 10960798, 9205044, 11758479,   8146043, 4281265, 8555166, 7887960, 11450315,   12647649, 7289348, 4379850, 5121118, 10662595,   5461275, 6431741, 10977468, 3561372, 8648759]

p = len(instrip + eclipse + sunlike + kdwarf + giant)
kidlist = [instrip, eclipse, sunlike, kdwarf, giant]
col=np.empty([p],dtype='S10')

training = np.empty([p,n])
pd = np.empty([p,n])
h = 0
for i in kidlist:
    for j in i:
        #print h, j
        temp = testdata[np.where(s.kepid==j)[0][0]]
        temp2 = plotdata[np.where(s.kepid==j)[0][0]]
        for k in range(0,n):
            training[h,k] = temp[k]
            pd[h,k] = temp2[k]
        if i==instrip: col[h] = 'blue'
        elif i==eclipse: col[h] = 'green'
        elif i==sunlike: col[h] = 'yellow'
        elif i==kdwarf: col[h] = 'orange'
        elif i==giant: col[h] = 'red'
        
        h = h+1

#q = len([x for x in testdata if x[0]>0])

'''
q=5000
i=0
j=0
sample = np.empty([q,n])
pd = np.empty([q,m])
col=np.empty([q],dtype='S10')
while i<len(sample):
    if testdata[j,0] > -10:
        sample[i] = testdata[j]
        pd[i] = plotdata[j]
        i=i+1
    j=j+1
'''
sample = training

sample2 = np.empty([len(sample),n])
for i in range(0,n):
    temp = sample[:,i].argsort()
    sample2[:,i] = np.arange(len(sample[:,i]))[temp.argsort()]

sample2 = sample2/len(sample)

print np.shape(sample)

#fig = pl.figure(1)
#pl.clf()

#col=np.empty([len(sample)],dtype='S10')

#i=0
#while i<len(sample):
'''
    if labels[i] == 0: col[i]='blue'
    if labels[i] == 1: col[i]='cyan'
    if labels[i] == 2: col[i]='green'
    if labels[i] == 3: col[i]='yellow'
    if labels[i] == 4: col[i]='orange'
    if labels[i] == 5: col[i]='red'
    if labels[i] == 6: col[i]='magenta'
    if labels[i] >= 7: col[i]='violet'
    if labels[i] == -1: col[i]='white'
'''
'''
    if 0<pd[i,1]<=(6.0-0.0004*pd[i,0]) and pd[i,1]<=4.0: col[i]='red'
    elif pd[i,1]>=4.2 and 5600<=pd[i,0]<=5900: col[i]='yellow'
    elif (pd[i,1]>=4.2 or pd[i,1]>=2.2+0.0005*pd[i,0]) and 3500<=pd[i,0]<=5100:
        col[i]='orange'
    else: col[i]='white'
'''
    #i=i+1

#colors = itertools.cycle('bgrcmybgrcmybgrcmybgrcmy')
#for k, col in zip(set(labels), colors):
#    for index in class_members:
#        x = testdata[:,index]
#        pl.plot(x[17], x[0], 'o', markerfacecolor=col,
#                markeredgecolor=col, markersize=1)
#        pl.plot(0,0)
#pl.show()

#i=1
#while i<=1:
#    clf = KMeans(n_clusters=i, max_iter=1, random_state=0)
#    clf.fit(sample)

#train = testdata[:10000]
#test = testdata[10000:20000]

#@pickle_results('forest.pkl')
#def compute_forest(depth):
    #rms_test = np.zeros(len(depth))
    #rms_train = np.zeros(len(depth))
    #i_best = 0

    #clf = RandomForestClassifier(n_estimators=1, max_depth=5,
    #                             min_samples_split=1, random_state=0)
    #stuff = clf.fit(rms_train)
    #stuff = clf.apply(rms_test)
    #print stuff

#compute_forest([0,1,2,3,4])

#    for i, d in enumerate(depth):
#        clf = RandomForestClassifier(n_estimators=10, max_depth=d, 
#                                     min_samples_split=1, random_state=0)
#        stuff = cross_val_score(clf, rms_test, rms_train)
#        print stuff

#map = PCA(n_components=3)
map = RandomizedPCA(n_components=3)
#map = LocallyLinearEmbedding(n_components=3, n_neighbors=15, method='standard')
#map = Isomap(n_components=3, n_neighbors=20)
X = map.fit_transform(sample)

fig = pl.figure(1)
pl.clf()
#print map.components_
#print map.explained_variance_ratio_
#print map.explained_variance_ratio_.sum()
#print X[0,0], X[0,1], X[0,2], col[0]

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

ax1 = pl.subplot(221, axisbg='black')
ax1.scatter(X[:,0],X[:,1],c=col,s=7,lw=0,picker=True)
fig.canvas.mpl_connect('pick_event',onpick)
pl.xlabel('component 1')
pl.ylabel('component 2')
'''
ax2 = pl.subplot(223, sharex=ax1, axisbg='black')
ax2.scatter(X[:,0],X[:,2],c=col,s=7,lw=0)
pl.xlabel('component 1')
pl.ylabel('component 3')

ax3 = pl.subplot(224, sharey=ax2, axisbg='black')
ax3.scatter(X[:,1],X[:,2],c=col,s=7,lw=0)
pl.xlabel('component 2')
'''
ax2 = pl.subplot(222,axisbg='black')
ax2.scatter(pd[:,0],pd[:,1],c=col,lw=0,s=7,picker=True)
#ax1.scatter(sample[:,0],sample[:,1],c=col,lw=0,s=7,picker=True)                
fig.canvas.mpl_connect('pick_event',onpick)
pl.xlim(2000,12000)
pl.ylim(0,6)
pl.xlabel('T_eff')
pl.ylabel('log(g)')
'''
ax2 = pl.subplot(223,axisbg='black')
ax2.scatter(pd[:,2],pd[:,3],c=col,lw=0,s=7,picker=True)
#ax2.scatter(sample[:,2],sample[:,3],c=col,lw=0,s=7,picker=True)                
fig.canvas.mpl_connect('pick_event',onpick)
pl.xlim(-1,4)
pl.ylim(-1,3.5)
pl.xlabel('log-first peak height')
pl.ylabel('log-second peak height')
'''
ax3 = pl.subplot(223,axisbg='black')
ax3.scatter(pd[:,4],pd[:,5],c=col,lw=0,s=7,picker=True)
#ax3.scatter(sample[:,4],sample[:,5],c=col,lw=0,s=7,picker=True)                
fig.canvas.mpl_connect('pick_event',onpick)
pl.xlim(-1.5,2.5)
pl.ylim(-1.5,2.5)
pl.xlabel('log-first peak period (d)')
pl.ylabel('log-second peak period (d)')

ax4 = pl.subplot(224,axisbg='black')
ax4.scatter(pd[:,6],pd[:,7],c=col,lw=0,s=7,picker=True)
#ax4.scatter(sample[:,6],sample[:,7],c=col,lw=0,s=7,picker=True)                
fig.canvas.mpl_connect('pick_event',onpick)
pl.xlim(-4,2)
pl.ylim(-3,3)
pl.xlabel('log-amplitude')
pl.ylabel('RMS deviation')

pl.show()
