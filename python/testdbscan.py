# runs DBSCAN on a sample with varying eps and min_samples
# to help fine tune them to produce a desired number of clusters and outliers

import numpy as np
import pyfits as pf
import itertools
from scipy.io.idl import readsav
from scipy.spatial import distance
import pylab as pl
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GMM
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from astroML.datasets import fetch_sdss_specgals
from astroML.decorators import pickle_results

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
fr = np.log10(s.maxflx/s.sndflx)
pr = np.log10(s.maxper/s.sndper)
hr = np.log10(s.maxpkht/s.sndht)
mp = np.log10(s.maxper)
sp = np.log10(s.sndper)
mh = np.log10(s.maxpkht)

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

#plotdata = [teff, s.logg, rad, amp, fr, pr, mw, sw, mp, sp, hr, mh, nmw, nsw]
#m=len(plotdata)
#plotdata = np.transpose(plotdata)
#testdata = [s.teff, s.logg, s.radius, s.range, htrat, s.maxpkht, perrat, s.sndht, nmw, nsw]           
testdata = [s.maxpkht, s.sndht, s.maxper, s.sndper, nmw2, nsw2, amp2]
n=len(testdata)
testdata = np.transpose(testdata)

i=0
j=0
sample = np.empty([5000,n])
while i<len(sample):
    if testdata[j,0] > -10:
        sample[i] = testdata[j]
        i=i+1
    j=j+1

#i=0
#while i<len(sample):
#    if sample[i,0] <= 0:
#        np.delete(sample,i)
#    else:
#        i=i+1

print np.shape(sample)
D = distance.squareform(distance.pdist(sample))
S = D/np.max(D)

clusters = np.empty([10,15])
for i in range (11,26):
    for j in range(1,11):
        db = DBSCAN(eps=i/10., min_samples=j).fit(S)
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        noise = 0
        for k in range(0,len(labels)):
            if labels[k] == -1: noise = noise+1
        clusters[j-1,i-11] = n_clusters_
        #clusters[j-1,i-1] = noise
        print i,j,n_clusters_,noise

#pl.figure(1)
#pl.clf()

#col=np.empty([len(sample)],dtype='S10')

#ax1 = pl.subplot(111,axisbg='black')
#ax1.hist2d(clusters,bins=40)
#pl.xlabel('Largest peak height')
#pl.ylabel('Second peak height')
#pl.xlim([0,2500])
#pl.ylim([0,40000])

pl.imshow(np.log10(clusters), interpolation='nearest')
pl.xlabel('eps x 10')
pl.ylabel('min_samples')
pl.colorbar()
pl.show()
