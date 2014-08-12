# supposed to plot the entire data set, color-coded by cuts in teff-logg space
# actually only plots about half of the points

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
#from __future__ import print_function

s = readsav('../data/grndsts9_vars.sav')

#maxwid = s.maxflx/s.maxpkht
#sndwid = s.sndflx/s.sndht
#mw = maxwid*1
#sw = sndwid*1
#nmw = np.log10(mw/s.maxper)
#nsw = np.log10(sw/s.sndper)
#nwrat = np.log10(mw/s.maxper*sw/s.sndper)
#fr = np.log10(s.maxflx/s.sndflx)
#pr = np.log10(s.maxper/s.sndper)
#hr = np.log10(s.maxpkht/s.sndht)
mp = np.log10(s.maxper)
sp = np.log10(s.sndper)
mh = np.log10(s.maxpkht)
sh = np.log10(s.sndht)

#flxrat = np.log10(s.maxflx/s.sndflx)
#perrat = np.log10(s.maxper/s.sndper)
#maxwid = (maxwid-np.mean(maxwid))/np.std(maxwid)
#sndwid = (sndwid-np.mean(sndwid))/np.std(sndwid)
#flxrat = (flxrat-np.mean(flxrat))/np.std(flxrat)
#perrat = (perrat-np.mean(perrat))/np.std(perrat)
#htrat = (hr-np.mean(hr))/np.std(hr)
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
        #s.radius[i] = np.log10(s.radius[i])
        s.range[i] = np.log10(s.range[i])
    if s.rms4[i] > 0:
        s.rms4[i] = np.log10(s.rms4[i])
    #if s.ncr[i] > 0:
        #s.ncr[i] = np.log10(s.ncr[i])
    #if s.ncrsm[i] > 0:
        #s.ncrsm[i] = np.log10(s.ncrsm[i])
    i=i+1

tm = np.mean([x for x in s.teff if x!=0])
ts = np.std([x for x in s.teff if x!=0])
#rm = np.mean([x for x in s.radius if x!=0])
#rs = np.std([x for x in s.radius if x!=0])
ampm = np.mean([x for x in s.range if x!=0])
amps = np.std([x for x in s.range if x!=0])
lm = np.mean([x for x in s.logg if x!=0])
ls = np.std([x for x in s.logg if x!=0])

s.maxpkht = (s.maxpkht-np.mean(s.maxpkht))/np.std(s.maxpkht)
s.maxper = (s.maxper-np.mean(s.maxper))/np.std(s.maxper)
#s.maxflx = (s.maxflx-np.mean(s.maxflx))/np.std(s.maxflx)
s.sndht = (s.sndht-np.mean(s.sndht))/np.std(s.sndht)
s.sndper = (s.sndper-np.mean(s.sndper))/np.std(s.sndper)
#s.sndflx = (s.sndflx-np.mean(s.sndflx))/np.std(s.sndflx)
teff2 = (s.teff-tm)/ts
logg2 = (s.logg-lm)/ls
#rad2 = (s.radius-rm)/rs
amp2 = (s.range-ampm)/amps
#nmw2 = (nmw-np.mean(nmw))/np.std(nmw)
#nsw2 = (nsw-np.mean(nsw))/np.std(nsw)
#nwrat = (nwrat-np.mean(nwrat))/np.std(nwrat)

rms4 = (s.rms4-np.mean(s.rms4))/np.std(s.rms4)
#ncr = (s.ncr-np.mean(s.ncr))/np.std(s.ncr)
#ncrsm = (s.ncrsm-np.mean(s.ncrsm))/np.std(s.ncrsm)

plotdata = [teff, s.logg, mh, sh, mp, sp, amp, r4]
m=len(plotdata)
plotdata = np.transpose(plotdata)
#testdata = [s.teff, s.logg, s.radius, s.range, htrat, s.maxpkht, perrat, s.sndht, nmw, nsw]
#testdata = [teff2, logg2, s.maxpkht, s.sndht, s.maxper, s.sndper, nmw2, nsw2, amp2]
testdata = [teff2, logg2, s.maxpkht, s.sndht, s.maxper, s.sndper, amp2, rms4]
n=len(testdata)
testdata = np.transpose(testdata)

#y = np.where(s.kepid==4241946)[0][0]
#print y, teff[y]

'''
i=0
j=0
while i<100:
    if 5600<=teff[j]<=5900 and s.logg[j]>=4.2:
    #if teff[j]>0 and s.logg[j]<=4.0:
        if i>=0: print s.kepid[j], teff[j], s.logg[j]
        i = i+1
    j = j+1
'''
# 4 pulsating, 100 eclipsing, 100 sun-like, 100 k-dwarfs, 100 giants
instrip = [2571868, 5356349, 5437206, 8489712]

eclipse = [8912468, 10855535, 9612468, 9898401, 7375612,   12350008, 6287172, 11825204, 4921906, 8288741,   12055255, 8108785, 1572353, 10288502, 9238207,   8555795, 6144827, 10030943, 10965091, 2715417,   11413213, 9077796, 6050116, 6350020, 7198474,   7546791, 3972629, 9345163, 8816790, 11494583,   4738426, 8122124, 9032671, 5960283, 9412114,   9004380, 4857282, 12602985, 11336707, 12104285,   11769739, 4138301, 9478836, 12508348, 6871716,   3839964, 12598713, 9700154, 9662581, 2856960,   7339345, 9288175, 9392331, 9239684, 11284547,   8045121, 10557008, 5785551, 9388303, 2437038,   7269843, 6072578, 9026766, 10802917, 3832382,   7697065, 9760531, 5956588, 2448320, 8739802,   4385109, 4563150, 5104097, 7680593, 3853259,   11566174, 3354616, 10074939, 2570289, 11246163,   4476900, 9508052, 9357030, 5786545, 9527167,   9179806, 7272739, 10600319, 5218385, 7269797,   5891963, 7199353, 10350225, 8242493, 4241946,   11405559, 8677949, 11036301, 7367833, 5459373]

sunlike = [9402544, 5640244, 11304194, 5369454, 2020086,   5878913, 10154417, 8866956, 10252521, 3648333,   10683445, 11034017, 11447613, 3118796, 8378779,   5097045, 9820618, 5521442, 4751659, 11124274,   5352130, 8572746, 8155776, 4557072, 3218513,   3655052, 11197005, 892977, 10149905, 11304779,   11241419, 5007471, 4252873, 7739563, 5375035,   8029017, 3229706, 4349736, 5734940, 8754254,   3663494, 10219365, 6590409, 7354587, 5695004,   8085053, 12356937, 11876270, 11407841, 4848497,   5611665, 10903132, 8415200, 9767803, 3348483,   8804764, 10677186, 7517664, 3218489, 12460038,   10027733, 11125962, 7948588, 8759831, 7265433,   10790476, 10924414, 11920065, 3865180, 7875139,   3341249, 10546672, 9007306, 11351200, 7756956,   5301750, 8957663, 9908470, 10815701, 7619004,   7950923, 6974841, 6431250, 10357182, 5872139,   3102541, 8197696, 5288099, 9702895, 9205138,   5031583, 7937360, 7848826, 8266042, 7661097,   11518142, 10854310, 9697090, 9592370, 12553011]

kdwarf = [2711088, 5481021, 6351151, 10908779, 7941412,   10556201, 10264852, 9512360, 9162025, 10963092,   4939907, 11020073, 7834941, 9722513, 9329586,   5094903, 4651312, 12102598, 11128748, 9569397,   4265713, 11401160, 11133844, 6151053, 7456580,   11145556, 12118164, 10273194, 7593903, 5784777,   8414716, 4730961, 4279048, 6945737, 3561525,   8980730, 3431530, 8331188, 8555903, 10256970,   10002597, 12166343, 11619032, 7362434, 8482464,   8226949, 12207283, 11751991, 10590402, 9112164,   11297936, 3234193, 2858337, 10748393, 10866057,   9307131, 11468870, 2304757, 4385594, 2574216,   10092136, 7591637, 5218109, 7938499, 5780269,   7691983, 1849846, 10284971, 5696108, 7538946,   3239770, 10487796, 10601075, 1571340, 10073680,   4937606, 10195818, 4665816, 12012281, 8604666,   11820922, 6862174, 9183248, 4569334, 8716714,   9031452, 8631563, 3728432, 11018253, 12401216,   8832146, 10452526, 4157460, 7907093, 5461988,   10983796, 3627577, 9674770, 3647136, 7987781]

giant = [4914151, 7971070, 11134982, 1431316, 6801204,   9726997, 8773948, 6358970, 8452840, 10685068,   8111555, 5802193, 5446242, 6877290, 11152894,   5396936, 9024768, 10801792, 11605089, 3943131,   4736439, 8398303, 8420916, 7696976, 6525397,   5523030, 3755364, 4481457, 11759700, 6524878,   8806387, 2581554, 11341730, 8804145, 2995050,   7120976, 6762447, 7630335, 10014893, 4656246,   1870028, 11911699, 8077790, 8256572, 7537808,   9368745, 6519747, 9782964, 8613617, 2574191,   11299524, 8936347, 5632912, 9532445, 10318539,   10908685, 2014377, 3962731, 2852181, 11392017,   11403918, 11710462, 7690810, 8848916, 11128654,   4383902, 5978547, 7265993, 9175366, 5450141,   4243819, 8824264, 11517599, 8365835, 10743331,   10162765, 7046845, 7377422, 8394974, 9182085,   9713434, 8782196, 4038624, 6213181, 6776331,   12602978, 8058062, 9699882, 9490342, 5018570,   11919968, 3356237, 11496193, 8086682, 7667124,   10881149, 8818614, 4068867, 10284878, 10811753]

p = len(instrip + eclipse + sunlike + kdwarf + giant)
kidlist = [instrip, eclipse, sunlike, kdwarf, giant]
#col=np.empty([len(testdata)],dtype='S10')
'''
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
'''
q = len([x for x in testdata if x[0]>0])

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

#sample = training

sample2 = np.empty([len(sample),n])
for i in range(0,n):
    temp = sample[:,i].argsort()
    sample2[:,i] = np.arange(len(sample[:,i]))[temp.argsort()]

sample2 = sample2/len(sample)

#i=0
#while i<len(sample):
#    if sample[i,0] <= 0:
#        np.delete(sample,i)
#    else:
#        i=i+1

print np.shape(sample)
#D = distance.squareform(distance.pdist(sample))
#S = 1 - (D/np.max(D))

#db = DBSCAN(eps=0.4, min_samples=5).fit(S)

#core_samples = db.core_sample_indices_
#labels = db.labels_
#components = db.components_
#n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

#print np.shape(testdata)
#print core_samples
#print labels[0:999]
#print labels[1000:1999]
#print n_clusters_
#print components

#noise = 0
#for k in range(0,len(labels)):
#    if labels[k] == -1: noise = noise+1

#print noise

#print "Silhouette Coefficient: %0.3f" % \
#    metrics.silhouette_score(D, labels, metric='precomputed')

fig = pl.figure(1)
pl.clf()
#pl.xscale('log')
#pl.yscale('log')
'''
i=0
j=0
labels2 = labels
while i<len(labels):
    #if labels[i]==0 or labels[i]==-1:
    if labels[i]==0:
        sample = np.delete(sample,j,0)
        sample2 = np.delete(sample2,j,0)
        labels2 = np.delete(labels2,j)
        pd = np.delete(pd,j,0)
    else:
        j = j+1
    i = i+1
labels = labels2
'''

#col=np.empty([len(sample)],dtype='S10')

i=0
while i<len(sample):
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
    if 0<pd[i,1]<=(6.0-0.0004*pd[i,0]) and pd[i,1]<=4.0: col[i]='red'
    elif pd[i,1]>=4.2 and 5600<=pd[i,0]<=5900: col[i]='yellow'
    elif (pd[i,1]>=4.2 or pd[i,1]>=2.2+0.0005*pd[i,0]) and 3500<=pd[i,0]<=5100:
        col[i]='orange'
    else: col[i]='white'
#    pl.scatter(sample[i,17], sample[i,0], c=col, lw=1)
    i=i+1
#pl.scatter(sample[0,17],sample[0,0],c='red', lw=1)

def onpick(event):
    ind=event.ind
    #for i in ind:
    #    print 'T_eff={0}, logg={1:.0f}'.format(int(pd[i,0]),int(10**pd[i,1]))
    #print
    s1 = ax1.scatter(pd[ind,0],pd[ind,1],c=col[ind],lw=1,s=36)
    s2 = ax2.scatter(pd[ind,2],pd[ind,3],c=col[ind],lw=1,s=36)
    s3 = ax3.scatter(pd[ind,4],pd[ind,5],c=col[ind],lw=1,s=36)
    s4 = ax4.scatter(pd[ind,6],pd[ind,7],c=col[ind],lw=1,s=36)
    '''s1 = ax1.scatter(sample[ind,0],sample[ind,1],c=col[ind],lw=1,s=36)
    s2 = ax2.scatter(sample[ind,2],sample[ind,3],c=col[ind],lw=1,s=36)
    s3 = ax3.scatter(sample[ind,4],sample[ind,5],c=col[ind],lw=1,s=36)
    s4 = ax4.scatter(sample[ind,6],sample[ind,7],c=col[ind],lw=1,s=36)
    s5 = ax5.scatter(sample2[ind,0],sample2[ind,1],c=col[ind],lw=1,s=36)
    s6 = ax6.scatter(sample2[ind,2],sample2[ind,3],c=col[ind],lw=1,s=36)
    s7 = ax7.scatter(sample2[ind,4],sample2[ind,5],c=col[ind],lw=1,s=36)
    s8 = ax8.scatter(sample2[ind,6],sample2[ind,7],c=col[ind],lw=1,s=36)'''
    #s9 = ax9.scatter(sample[ind,4],sample[ind,8],c=col[ind],lw=1,s=36)
    #s10 = ax10.scatter(sample2[ind,4],sample2[ind,8],c=col[ind],lw=1,s=36)
    pl.ion()
    pl.draw()
    s1.set_visible(False)
    s2.set_visible(False)
    s3.set_visible(False)
    s4.set_visible(False)
    '''s5.set_visible(False)
    s6.set_visible(False)
    s7.set_visible(False)
    s8.set_visible(False)'''

ax1 = pl.subplot(221,axisbg='black')
ax1.scatter(pd[:1000,0],pd[:1000,1],c=col,lw=0,s=7,picker=True)
#ax1.scatter(sample[:,0],sample[:,1],c=col,lw=0,s=7,picker=True)
fig.canvas.mpl_connect('pick_event',onpick)
pl.xlabel('T_eff')
pl.ylabel('log(g)')

ax2 = pl.subplot(222,axisbg='black')
ax2.scatter(pd[:1000,2],pd[:1000,3],c=col,lw=0,s=7,picker=True)
#ax2.scatter(sample[:,2],sample[:,3],c=col,lw=0,s=7,picker=True)
fig.canvas.mpl_connect('pick_event',onpick)
pl.xlabel('log-first peak height')
pl.ylabel('log-second peak height')

ax3 = pl.subplot(223,axisbg='black')
ax3.scatter(pd[:1000,4],pd[:1000,5],c=col,lw=0,s=7,picker=True)
#ax3.scatter(sample[:,4],sample[:,5],c=col,lw=0,s=7,picker=True)
fig.canvas.mpl_connect('pick_event',onpick)
pl.xlabel('log-first peak period (d)')
pl.ylabel('log-second peak period (d)')

ax4 = pl.subplot(224,axisbg='black')
ax4.scatter(pd[:1000,6],pd[:1000,7],c=col,lw=0,s=7,picker=True)
#ax4.scatter(sample[:,6],sample[:,7],c=col,lw=0,s=7,picker=True)
fig.canvas.mpl_connect('pick_event',onpick)
pl.xlabel('log-amplitude')
pl.ylabel('RMS deviation')
'''
ax5 = pl.subplot(245,axisbg='black')
#ax1.scatter(pd[:,0],pd[:,1],c=col,lw=0,s=7,picker=True)
ax5.scatter(sample2[:,0],sample2[:,1],c=col,lw=0,s=7,picker=True)
fig.canvas.mpl_connect('pick_event',onpick)

ax6 = pl.subplot(246,axisbg='black')
#ax2.scatter(pd[:,6],pd[:,2],c=col,lw=0,s=7,picker=True)
ax6.scatter(sample2[:,2],sample2[:,3],c=col,lw=0,s=7,picker=True)
fig.canvas.mpl_connect('pick_event',onpick)

ax7 = pl.subplot(247,axisbg='black')
#ax3.scatter(pd[:,9],pd[:,10],c=col,lw=0,s=7,picker=True)
ax7.scatter(sample2[:,4],sample2[:,5],c=col,lw=0,s=7,picker=True)
fig.canvas.mpl_connect('pick_event',onpick)

ax8 = pl.subplot(248,axisbg='black')
#ax4.scatter(pd[:,4],pd[:,5],c=col,lw=0,s=7,picker=True)
ax8.scatter(sample2[:,6],sample2[:,7],c=col,lw=0,s=7,picker=True)
fig.canvas.mpl_connect('pick_event',onpick)

ax9 = pl.subplot(255,axisbg='black')
#ax3.scatter(pd[:,9],pd[:,10],c=col,lw=0,s=7,picker=True)
ax9.scatter(sample[:,4],sample[:,8],c=col,lw=0,s=7,picker=True)
fig.canvas.mpl_connect('pick_event',onpick)
pl.xlabel('First peak period')
pl.ylabel('Amplitude')

ax10 = pl.subplot(2,5,10,axisbg='black')
#ax4.scatter(pd[:,4],pd[:,5],c=col,lw=0,s=7,picker=True)
ax10.scatter(sample2[:,4],sample2[:,8],c=col,lw=0,s=7,picker=True)
fig.canvas.mpl_connect('pick_event',onpick)
'''
pl.show()

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

#pl.figure(figsize=(16,10))
#ax1 = pl.subplot(221)
#pl.scatter(pca.components_[0],pca.components_[1],s=0.1,lw=0)
#pl.ylabel('component 2')

#ax2 = pl.subplot(223, sharex=ax1)
#pl.scatter(pca.components_[0],pca.components_[2],s=0.1,lw=0)
#pl.xlabel('component 1')
#pl.ylabel('component 3')

#ax3 = pl.subplot(224, sharey=ax2)
#pl.scatter(pca.components_[1],pca.components_[2],s=0.1,lw=0)
#pl.xlabel('component 2')

#pl.show()
