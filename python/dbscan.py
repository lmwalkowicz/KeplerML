# runs DBSCAN on a sample of points and plots the results

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

#s.maxpkht = s.maxpkht/np.median(s.maxpkht)
#s.maxper = s.maxper/np.median(s.maxper)
#s.maxflx = s.maxflx/np.median(s.maxflx)
#s.sndht = s.sndht/np.median(s.sndht)
#s.sndper = s.sndper/np.median(s.sndper)
#s.sndflx = s.sndflx/np.median(s.sndflx)

maxwid = s.maxflx/s.maxpkht
sndwid = s.sndflx/s.sndht
mw = maxwid*1
sw = sndwid*1
nmw = np.log10(mw/s.maxper)
nsw = np.log10(sw/s.sndper)
nwrat = np.log10(mw/s.maxper*sw/s.sndper)
fr = np.log10(s.maxflx/s.sndflx)
pr = np.log10(s.maxper/s.sndper)
hr = np.log10(s.maxpkht/s.sndht)
mp = np.log10(s.maxper)
sp = np.log10(s.sndper)
mh = np.log10(s.maxpkht)
sh = np.log10(s.sndht)

flxrat = np.log10(s.maxflx/s.sndflx)
perrat = np.log10(s.maxper/s.sndper)
maxwid = (maxwid-np.mean(maxwid))/np.std(maxwid)
sndwid = (sndwid-np.mean(sndwid))/np.std(sndwid)
flxrat = (flxrat-np.mean(flxrat))/np.std(flxrat)
perrat = (perrat-np.mean(perrat))/np.std(perrat)
htrat = (hr-np.mean(hr))/np.std(hr)
s.maxper = np.log10(s.maxper)
s.sndper = np.log10(s.sndper)
s.maxpkht = np.log10(s.maxpkht)
s.sndht = np.log10(s.sndht)
s.range = s.range/s.totf

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
        s.radius[i] = np.log10(s.radius[i])
        s.range[i] = np.log10(s.range[i])
    i=i+1

tm = np.mean([x for x in s.teff if x!=0])
ts = np.std([x for x in s.teff if x!=0])
rm = np.mean([x for x in s.radius if x!=0])
rs = np.std([x for x in s.radius if x!=0])
ampm = np.mean([x for x in s.range if x!=0])
amps = np.std([x for x in s.range if x!=0])
lm = np.mean([x for x in s.logg if x!=0])
ls = np.std([x for x in s.logg if x!=0])

s.maxpkht = (s.maxpkht-np.mean(s.maxpkht))/np.std(s.maxpkht)
s.maxper = (s.maxper-np.mean(s.maxper))/np.std(s.maxper)
s.maxflx = (s.maxflx-np.mean(s.maxflx))/np.std(s.maxflx)
s.sndht = (s.sndht-np.mean(s.sndht))/np.std(s.sndht)
s.sndper = (s.sndper-np.mean(s.sndper))/np.std(s.sndper)
s.sndflx = (s.sndflx-np.mean(s.sndflx))/np.std(s.sndflx)
teff2 = (s.teff-tm)/ts
logg2 = (s.logg-lm)/ls
rad2 = (s.radius-rm)/rs
amp2 = (s.range-ampm)/amps
nmw2 = (nmw-np.mean(nmw))/np.std(nmw)
nsw2 = (nsw-np.mean(nsw))/np.std(nsw)
nwrat = (nwrat-np.mean(nwrat))/np.std(nwrat)

plotdata = [teff, rad, amp, pr, mp, sp, hr, mh, sh, nmw, nsw]
m=len(plotdata)
plotdata = np.transpose(plotdata)
#testdata = [s.teff, s.logg, s.radius, s.range, htrat, s.maxpkht, perrat, s.sndht, nmw, nsw]
#testdata = [teff2, logg2, s.maxpkht, s.sndht, s.maxper, s.sndper, nmw2, nsw2, amp2]
testdata = [teff2, logg2, s.maxpkht, s.sndht, s.maxper, s.sndper, amp2]
n=len(testdata)
testdata = np.transpose(testdata)

i=0
j=0
sample = np.empty([5000,n])
pd = np.empty([5000,m])
while i<len(sample):
    if testdata[j,0] > -10:
        sample[i] = testdata[j]
        pd[i] = plotdata[j]
        i=i+1
    j=j+1

sample2 = np.empty([5000,n])
for i in range(0,n):
    temp = sample[:,i].argsort()
    sample2[:,i] = np.arange(len(sample[:,i]))[temp.argsort()]

sample2 = sample2/5000

#i=0
#while i<len(sample):
#    if sample[i,0] <= 0:
#        np.delete(sample,i)
#    else:
#        i=i+1

print np.shape(sample)
D = distance.squareform(distance.pdist(sample))
S = 1 - (D/np.max(D))

db = DBSCAN(eps=1.5, min_samples=5).fit(S)

core_samples = db.core_sample_indices_
labels = db.labels_
components = db.components_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

#print np.shape(testdata)
#print core_samples
#print labels[0:999]
#print labels[1000:1999]
print n_clusters_
#print components

noise = 0
for k in range(0,len(labels)):
    if labels[k] == -1: noise = noise+1

print noise

print "Silhouette Coefficient: %0.3f" % \
    metrics.silhouette_score(D, labels, metric='precomputed')

fig = pl.figure(1)
pl.clf()
#pl.xscale('log')
#pl.yscale('log')

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

col=np.empty([len(sample)],dtype='S10')

i=0
while i<len(sample):
#    print labels[i]
    if labels[i] == 0: col[i]='blue'
    if labels[i] == 1: col[i]='cyan'
    if labels[i] == 2: col[i]='green'
    if labels[i] == 3: col[i]='yellow'
    if labels[i] == 4: col[i]='orange'
    if labels[i] == 5: col[i]='red'
    if labels[i] == 6: col[i]='magenta'
    if labels[i] >= 7: col[i]='violet'
    if labels[i] == -1: col[i]='white'
#    print col
#    pl.scatter(sample[i,17], sample[i,0], c=col, lw=1)
    i=i+1
#pl.scatter(sample[0,17],sample[0,0],c='red', lw=1)

def onpick(event):
    ind=event.ind
    #for i in ind:
    #    print 'T_eff={0}, logg={1:.0f}'.format(int(pd[i,0]),int(10**pd[i,1]))
    #print
    '''s1 = ax1.scatter(pd[ind,0],pd[ind,1],c=col[ind],lw=1,s=36)
    s2 = ax2.scatter(pd[ind,6],pd[ind,2],c=col[ind],lw=1,s=36)
    s3 = ax3.scatter(pd[ind,9],pd[ind,10],c=col[ind],lw=1,s=36)
    s4 = ax4.scatter(pd[ind,4],pd[ind,5],c=col[ind],lw=1,s=36)'''
    s1 = ax1.scatter(sample[ind,0],sample[ind,1],c=col[ind],lw=1,s=36)
    s2 = ax2.scatter(sample[ind,2],sample[ind,3],c=col[ind],lw=1,s=36)
    s3 = ax3.scatter(sample[ind,4],sample[ind,5],c=col[ind],lw=1,s=36)
    s4 = ax4.scatter(sample[ind,4],sample[ind,6],c=col[ind],lw=1,s=36)
    pl.ion()
    pl.draw()
    s1.set_visible(False)
    s2.set_visible(False)
    s3.set_visible(False)
    s4.set_visible(False)

ax1 = pl.subplot(221,axisbg='black')
#ax1.scatter(pd[:,0],pd[:,1],c=col,lw=0,s=7,picker=True)
ax1.scatter(sample[:,0],sample[:,1],c=col,lw=0,s=7,picker=True)
fig.canvas.mpl_connect('pick_event',onpick)
#pl.xlabel('T_eff')
#pl.ylabel('log Radius')
#pl.xlim([0,2500])
#pl.ylim([0,40000])

ax2 = pl.subplot(222,axisbg='black')
#ax2.scatter(pd[:,6],pd[:,2],c=col,lw=0,s=7,picker=True)
ax2.scatter(sample[:,2],sample[:,3],c=col,lw=0,s=7,picker=True)
fig.canvas.mpl_connect('pick_event',onpick)
#pl.xlabel('log Peak Height Ratio')
#pl.ylabel('log Normalized Amplitude')
#pl.xlim([0,40000])
#pl.ylim([0,20000])

ax3 = pl.subplot(223,axisbg='black')
#ax3.scatter(pd[:,9],pd[:,10],c=col,lw=0,s=7,picker=True)
ax3.scatter(sample[:,4],sample[:,5],c=col,lw=0,s=7,picker=True)
fig.canvas.mpl_connect('pick_event',onpick)
#pl.xlabel('Normed first peak width')
#pl.ylabel('Normed second peak width')
#pl.xlim([0,1200])
#pl.ylim([0,20000])

ax4 = pl.subplot(224,axisbg='black')
#ax4.scatter(pd[:,4],pd[:,5],c=col,lw=0,s=7,picker=True)
ax4.scatter(sample[:,4],sample[:,6],c=col,lw=0,s=7,picker=True)
fig.canvas.mpl_connect('pick_event',onpick)
#pl.xlabel('First peak period')
#pl.ylabel('Second peak period')

#ax5 = pl.subplot(335,axisbg='black')
#ax5.scatter(sample[:,4],sample[:,6],c=col,lw=0,s=7)
#ax6 = pl.subplot(336,axisbg='black')
#ax6.scatter(sample[:,4],sample[:,7],c=col,lw=0,s=7)
#ax7 = pl.subplot(337,axisbg='black')
#ax7.scatter(sample[:,5],sample[:,6],c=col,lw=0,s=7)
#ax8 = pl.subplot(338,axisbg='black')
#ax8.scatter(sample[:,5],sample[:,7],c=col,lw=0,s=7)
#ax9 = pl.subplot(339,axisbg='black')
#ax9.scatter(sample[:,6],sample[:,7],c=col,lw=0,s=7)

pl.show()



'''class DataCursor(object):
    text_template = 'x'
    x,y = 0.0,0.0
    xoffset,yoffset = -1,1
    text_template = T: %s\ng:%s

    def __init__(self, ax):
        self.ax = ax
        self.annotation = ax.annotate(self.text_template,
                                      xy=(self.x,self.y),xytext(0,0),
                                      textcoords='axes fraction',ha='left',
                                      va='bottom',fontsize=10,
                                      bbox-dict(boxstyle='round,pad=0.5',fc='yellow',alpha=1),
                                      arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=0'))
        self.annotation.set_visible(False)
        self.annotation.draggable()

    def __call__(self,event):
        self.event = event
        self.x,self.y = event.mouseevent.xdata,event.mouseevent.ydata
        if self.x is not None:
            glim=pickle.load(open())
            tlim=pickle.load(open())
            vlim=pickle.load(open())
            a=glim[event.ind[0]]
            b=tlim[event.ind[0]]
            c=vlim[event.ind[0]]
            temp_temp=self.text_template % (x,y))
            if temp_temp==self.annotation.get_text() and self.annotation.get_visible():
                self.annotation.set_visible(False)
                event.canvas.draw()
                return
            self.annotation.xy=self.x,self.y
            self.annotation.set_text(self.text_template % (x,y))
            self.annotation.set_visible(True)
            event.canvas.draw()'''

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
