#test
from __future__ import division
import random
import numpy as np
import scipy as sp
from scipy import stats
import pyfits
import math
import pylab as pl
import matplotlib.pyplot as plt
import heapq
from operator import xor
import scipy.signal
from numpy import float64
import astroML.time_series
import astroML_addons.periodogram
import cython
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN,Ward
from numpy.random import RandomState
rng = RandomState(42)
import itertools

"""cimport numpy as np
cimport cython

__all__ = ['lombscargle']


cdef extern from "math.h":
    double cos(double)
    double sin(double)
    double atan(double)

@cython.boundscheck(False)
def lombscargle(np.ndarray[np.float64_t, ndim=1] x,
                np.ndarray[np.float64_t, ndim=1] y,
                np.ndarray[np.float64_t, ndim=1] freqs):
"""

"""slope of trough vs slope of normal, or trough roundness/peak roundness or trough outliers/peak outliers"""

with open('infiles','r') as f:
	myarray=[line.split() for line in f]
print len(myarray)
test=[]
time=[]
err=[]
teff=[]

for i in range (63000, 64000):
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
MAIN TV 
for i in range(len(numpdc)):
	plt.scatter(numtime[i],numpdc[i],color='red')
	plt.show()
"""
"""2d ml plot""" 
#comment-see 54000 to 56000 for round outliers

def plot_2D(data, target, target_names):
	colors = itertools.cycle('rgbcmykw')
	target_ids = range(len(target_names))
	pl.figure()
	for i, c, label in zip(target_ids, colors, target_names):
		pl.scatter(data[target == i, 0], data[target == i, 1],c=c, label=label)	
	pl.legend()
	pl.show()

"""TOOLS
http://www.astro.princeton.edu/~jhartman/vartools.html

"""

"""--------------------------------PRELIMINARY CALCULATIONS-----------------------------------------"""
"""long term trend (slope)"""

longtermtrend=np.zeros(len(numpdc))

for i in range(len(numpdc)):
     longtermtrend[i]=np.polyfit(numtime[i], numpdc[i], 1)[0]

meanltt=np.mean(longtermtrend)
medianltt=np.median(longtermtrend)

"""y offset"""

y_offset=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	y_offset[i]=np.polyfit(numtime[i], numpdc[i], 1)[1]



"""Corr PDC-shifted downwards quite a bit so be careful"""
corrpdc=[0]*len(numpdc)
 
for i in range(len(numpdc)):
	corrpdc[i]=[(numpdc[i][j]-longtermtrend[i]*numtime[i][j]-y_offset[i]) for j in range(len(numpdc[i]))]


"""---------------------------------STANDARD STATS----------------------------------------"""




"""asymmetry measure here"""



"""mean and median flux and relationship"""

means=np.zeros(len(numpdc))  

for i in range(len(numpdc)):
	means[i]=np.mean(numpdc[i])

medians=np.zeros(len(numpdc))  

for i in range(len(numpdc)):
	medians[i]=np.median(numpdc[i])

meanmedrat=[means[i]/medians[i] for i in range(len(numpdc))]



"""skew"""
skews=np.zeros(len(numpdc))  

for i in range(len(numpdc)):
	skews[i]=scipy.stats.skew(numpdc[i])

"""Variance"""

varss=np.zeros(len(numpdc))  

for i in range(len(numpdc)):
	varss[i]=np.var(numpdc[i])

"""Coeff of variability"""

coeffvar=np.zeros(len(numpdc))  

for i in range(len(numpdc)):
	coeffvar[i]=np.std(numpdc[i])/np.mean(numpdc[i])



"""STD"""
stds=np.zeros(len(numpdc))  

for i in range(len(numpdc)):
	stds[i]=np.std(numpdc[i])

"""outliers beyond 4 sigma-decent clutering-also try one weighted according to n sigma"""

outliers=[0]*len(numpdc)
for i in range(len(numpdc)):
	outliers[i]=[numpdc[i][j] for j in range (len(numpdc[i])) if (numpdc[i][j]>means[i]+4*stds[i]) or (numpdc[i][j]<means[i]-4*stds[i])]  

numoutliers=np.zeros(len(numpdc))

for i in range(len(numpdc)):
	numoutliers[i]=len(outliers[i])

"""these outliers below for transits"""

negoutliers=[0]*len(numpdc)
for i in range(len(numpdc)):
	negoutliers[i]=[numpdc[i][j] for j in range (len(numpdc[i])) if (numpdc[i][j]<means[i]-4*stds[i])]  

numnegoutliers=np.zeros(len(numpdc))

for i in range(len(numpdc)):
	numnegoutliers[i]=len(negoutliers[i])

numposoutliers=[numoutliers[i]-numnegoutliers[i] for i in range(len(numpdc))]

"""beyond 1,2,3 std"""

out1std=[0]*len(numpdc)
out2std=[0]*len(numpdc)
out3std=[0]*len(numpdc)
for i in range(len(numpdc)):
	out1std[i]=[numpdc[i][j] for j in range (len(numpdc[i])) if (numpdc[i][j]>means[i]+stds[i]) or (numpdc[i][j]<means[i]-stds[i])]  
	"""out2std[i]=[numpdc[i][j] for j in range (len(numpdc[i])) if (numpdc[i][j]>means[i]+2*stds[i]) or (numpdc[i][j]<means[i]-2*stds[i])] 
	out3std[i]=[numpdc[i][j] for j in range (len(numpdc[i])) if (numpdc[i][j]>means[i]+3*stds[i]) or (numpdc[i][j]<means[i]-3*stds[i])] """

numout1s=np.zeros(len(numpdc))
numout2s=np.zeros(len(numpdc))
numout3s=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	numout1s[i]=len(out1std[i])
	"""numout2s[i]=len(out2std[i])
	numout3s[i]=len(out3std[i])"""

"""kurtosis"""

kurt=np.zeros(len(numpdc))

for i in range(len(numpdc)):
	kurt[i]=scipy.stats.kurtosis(numpdc[i])

"""Median AD (MAD)"""

mad=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	mad[i]=np.median([abs(numpdc[i][j]-medians[i]) for j in range(len(numpdc[i]))])

print 'check speed 1'

"""------------------------------------------------------------------------------------"""





"""------------------------------------------------SLOPES---------------------------------"""

"""slopes"""
slopes=[0]*(len(numpdc))
for i in range(len(numpdc)):    
	slopes[i]=[(numpdc[i][j+1]-numpdc[i][j])/(numtime[i][j+1]-numtime[i][j]) for j in range (len(numpdc[i])-1)]



"""mean slope- long term trend """
meanslope=np.zeros(len(numpdc))

for i in range(len(numpdc)):
	meanslope[i]=np.mean(slopes[i])


"""max and min slopes"""
maxslope=np.zeros(len(numpdc))
minslope=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	maxslope[i]=np.percentile(slopes[i],99)
	minslope[i]=np.percentile(slopes[i],1)

"""Corrected Slopes"""

corrslopes=[0]*(len(corrpdc))
for i in range(len(corrpdc)):    
	corrslopes[i]=[(corrpdc[i][j+1]-corrpdc[i][j])/(numtime[i][j+1]-numtime[i][j]) for j in range (len(corrpdc[i])-1)]

"""corrected mean slope here"""


"""ratio of mean pslope to mean nslope-general assymetry?-good clustering"""

pslope=[0]*(len(numpdc))

nslope=[0]*(len(numpdc))

for i in range(len(numpdc)):    
	pslope[i]=[corrslopes[i][j] for j in range (len(corrslopes[i])) if corrslopes[i][j]>=0]

for i in range(len(numpdc)):    
	nslope[i]=[corrslopes[i][j] for j in range (len(corrslopes[i])) if corrslopes[i][j]<0]

meanpslope=np.zeros(len(numpdc))
meannslope=np.zeros(len(numpdc))
g_asymm=np.zeros(len(numpdc))
rough_g_asymm=np.zeros(len(numpdc))
diff_asymm=np.zeros(len(numpdc))

for i in range(len(numpdc)):
	meanpslope[i]=np.mean(pslope[i])
	meannslope[i]=-np.mean(nslope[i])
	g_asymm[i]=meanpslope[i]/meannslope[i]
	rough_g_asymm[i]=len(pslope[i])/len(nslope[i])
	diff_asymm[i]=meanpslope[i]-meannslope[i]

"""skew slope- hope of asymmetry"""
skewslope=np.zeros(len(numpdc))  

for i in range(len(numpdc)):
	skewslope[i]=scipy.stats.skew(corrslopes[i])

"""plt.scatter(skewslope, diff_asymm)
plt.show()"""
"""special asymmetry"""

"""Abs slopes"""

absslopes=[0]*len(numpdc)

for i in range(len(numpdc)):
	absslopes[i]= [abs(corrslopes[i][j]) for j in range(len(corrslopes[i]))]

"""varabsslope"""

varabsslope=np.zeros(len(numpdc))
meanabsslope=np.zeros(len(numpdc))

meanabsslope=[np.var(absslopes[i]) for i in range(len(numpdc))]
varabsslope=[np.mean(absslopes[i]) for i in range(len(numpdc))]
testa=np.zeros(20)
testb=np.ones(20)
testa[10:19]=20
testb[10:19]=21




"""plot_2D(x_pca,dbscan.labels_, [i for i in range n_clusters])"""


"""
merge=np.vstack((varabsslope, g_asymm))
merge=merge.T
dbscan = DBSCAN(eps=200000000, min_samples=3,metric='euclidean').fit_predict(merge)
print dbscan
"""
"""print dbscan.labels_"""


"""var slope"""

varslope=np.zeros(len(numpdc))

varslope=[np.var(slopes[i]) for i in range(len(slopes))]

print 'check 1.5'

"""secders"""
secder=[0]*len(numpdc)

for i in range(len(numpdc)):
	secder[i]=[(slopes[i][j]-slopes[i][j-1])/((numtime[i][j+1]-numtime[i][j])/2+(numtime[i][j]-numtime[i][j-1])/2) for j in range (1, len(numpdc[i])-1)]

meansecder=np.zeros(len(numpdc))

for i in range(len(numpdc)):
		meansecder[i]=np.mean(secder[i])

abssecder=[0]*(len(numpdc))

for i in range(len(numpdc)):
 	abssecder[i]=[abs((slopes[i][j]-slopes[i][j-1])/((numtime[i][j+1]-numtime[i][j])/2+(numtime[i][j]-numtime[i][j-1])/2)) for j in range (1, len(slopes[i])-1)]

absmeansecder=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	absmeansecder[i]=np.mean(abssecder[i])

"""
plt.scatter(np.log(absmeansecder),skewslope)
plt.show()
plt.scatter(diff_asymm,g_asymm)
plt.show()
plt.scatter(maxslope,g_asymm)
plt.show()
plt.scatter(maxslope,np.log(absmeansecder))
plt.show()
plt.scatter(meanmedrat,np.log(absmeansecder))
plt.show()
plt.scatter(meanmedrat,g_asymm)
plt.show()
"""
"""corrsecders"""
corrsecder=[0]*len(numpdc)

for i in range(len(numpdc)):
	corrsecder[i]=[(corrslopes[i][j]-corrslopes[i][j-1])/((numtime[i][j+1]-numtime[i][j])/2+(numtime[i][j]-numtime[i][j-1])/2) for j in range (1, len(corrpdc[i])-1)]

"""as regards periodicity in general,there can exist many levels"""
"""Num_spikes- you casn also isolate transits from cataclysmics using periodicity of spikes
take ratios of roundnessess or multiply them, """
pspikes=[0]*len(numpdc)
nspikes=[0]*len(numpdc)
sdspikes=[0]*len(numpdc)
sdspikes2=[0]*len(numpdc)
pslopestds=np.zeros(len(numpdc))
nslopestds=np.zeros(len(numpdc))
sdstds=np.zeros(len(numpdc))
meanstds=np.zeros(len(numpdc))
stdratio=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	pslopestds[i]=np.std(pslope[i])
	nslopestds[i]=np.std(nslope[i])
	sdstds[i]=np.std(corrsecder[i])
	meanstds[i]=np.mean(corrsecder[i])
	stdratio[i]=pslopestds[i]/nslopestds[i]
"""
for i in range(len(numpdc)):
	pspikes[i]=[corrslopes[i][j] for j in range(len(corrslopes[i])) if corrslopes[i][j]>=3*slopestds[i]] 
	nspikes[i]=[corrslopes[i][j] for j in range(len(corrslopes[i])) if corrslopes[i][j]<=3*slopestds[i]]
	sdspikes[i]=[corrsecder[i][j] for j in range(len(corrsecder[i])) if corrsecder[i][j]>=4*sdstds[i]] 
"""
for i in range(len(numpdc)):
	pspikes[i]=[corrslopes[i][j] for j in range(len(corrslopes[i])) if corrslopes[i][j]>=meanpslope[i]+3*pslopestds[i]] 
	nspikes[i]=[corrslopes[i][j] for j in range(len(corrslopes[i])) if corrslopes[i][j]<=meannslope[i]-3*nslopestds[i]]
	sdspikes[i]=[corrsecder[i][j] for j in range(len(corrsecder[i])) if corrsecder[i][j]>=4*sdstds[i]] 
	sdspikes2[i]=[corrsecder[i][j] for j in range(len(corrsecder[i])) if corrsecder[i][j]<=-4*sdstds[i]]

"""change around the 4 and add the min condition along with sdspike
to look for transits"""

num_pspikes=np.zeros(len(numpdc))
num_nspikes=np.zeros(len(numpdc))
num_sdspikes=np.zeros(len(numpdc))
num_sdspikes2=np.zeros(len(numpdc))

for i in range(len(numpdc)):
	num_pspikes[i]=len(pspikes[i]) 
	num_nspikes[i]=len(nspikes[i])
	num_sdspikes[i]=len(sdspikes[i])
	num_sdspikes2[i]=len(sdspikes2[i])

"""pair slope trend"""
pstrend=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	pstrend[i]=len([slopes[i][j] for j in range(len(slopes[i])-1) if (slopes[i][j]>0) & (slopes[i][j+1]>0)])/len(slopes[i])

"""
print 'sdspikes'

for i in range(len(numpdc)):
	if num_sdspikes[i]>2*np.std(num_sdspikes):
		print i
		plt.scatter(numtime[i],numpdc[i])
		plt.show()
		plt.xlim(970,975)
		plt.scatter(numtime[i],numpdc[i])
		plt.show()
gdors have positive roundness so these combined will differentiate the two and gasymm and skews will be enough to differentiate
rrlyra and gdors from transits

print 'num_nspikes'
for i in range(len(numpdc)):
	if num_nspikes[i]>2*np.std(num_nspikes):
		print i
		print num_nspikes[i]
		plt.scatter(numtime[i],numpdc[i])
		plt.show()
		plt.xlim(970,975)
		plt.scatter(numtime[i],numpdc[i])
		plt.show()
		plt.xlim(971,972)
		plt.scatter(numtime[i],numpdc[i])
		plt.show()
print 'num_pspikes'
for i in range(len(numpdc)):
	if num_pspikes[i]>2*np.std(num_pspikes):
		print i
		print num_pspikes[i]
		plt.scatter(numtime[i],numpdc[i])
		plt.show()
		plt.xlim(970,975)
		plt.scatter(numtime[i],numpdc[i])
		plt.show()
		plt.xlim(971,972)
		plt.scatter(numtime[i],numpdc[i])
		plt.show()


for i in range(len(numpdc)):
	num_pspikes[i]=len([corrslopes[i][j] for j in range(len(corrslopes[i])) if corrslopes[i][j]>=3*np.std(corrslopes[i])]) 
	num_nspikes[i]=len([corrslopes[i][j] for j in range(len(corrslopes[i])) if corrslopes[i][j]<=3*np.std(corrslopes[i])])
	num_sdspikes[i]=len([corrsecder[i][j] for j in range(len(corrsecder[i])) if corrsecder[i][j]>=4*np.std(corrsecder[i])]) 
"""


print 'check speed 2'
"""---------------------------------------------------------------------------------"""



"""Zero crossings- accounted for ltt, plot with gasymm"""

zcrossind=[]
for i in range(len(numpdc)):
	ltt=longtermtrend[i]
	yoff=y_offset[i]
	zcrossind.append([j for j in range(len(numpdc[i])-1) if (ltt*numtime[i][j+1]+ yoff-numpdc[i][j+1])*(ltt*numtime[i][j]+yoff-numpdc[i][j])<0])


num_zcross=np.zeros(len(numpdc))

for i in range(len(numpdc)):
	num_zcross[i]=len(zcrossind[i])





"""plt.scatter(numtime[np.argmin(num_zcross)],numpdc[np.argmin(num_zcross)])
plt.show()"""

"""SLIDING CALCULATIONS"""

"""
Sliding mean, variance, zcrossings etc

slidingmean=[]

for i in range(len(numpdc)):
	slidingmean.append([])
	for j in range(len(numpdc[i])):
		slidingmean[i].append(np.mean(numpdc[i][max(j-10,0):min(j+10, len(numpdc[i])-1): 1]))

slidingvar=[]

for i in range(len(numpdc)):
	slidingvar.append([])
	for j in range(len(numpdc[i])):
		slidingvar[i].append(np.var(corrpdc[i][max(j-10,0):min(j+10, len(numpdc[i])-1): 1]))
changevar=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	changevar[i]=np.var(slidingvar[i])

most for gdors?

slidingzc=[]
changezc=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	slidingzc.append([])
	for j in range(len(numpdc[i])):
		slidingzc[i].append(len([k for k in range(max(j-30,0),min(j+30, len(numpdc[i])-1)) if k in zcrossind[i]]))

for i in range(len(numpdc)):
	changezc[i]=np.var(slidingzc[i])

print 'changevar,changezc'
plt.scatter(changevar,changezc)
pl.show()

"""

"""------------------------------------PEAKS AND TROUGHS---------------------------------------"""



"""pm"""
plusminus=[0]*len(numpdc)


for i in range(len(numpdc)):
	plusminus[i]=[j for j in range(1,len(slopes[i])) if (slopes[i][j]<0)&(slopes[i][j-1]>0)]

num_pm=np.zeros(len(numpdc))  
num_pm=[len(plusminus[i]) for i in range(len(numpdc))] 


"""mp

minusplus=[0]*len(numpdc)
for i in range(len(numpdc)):
	minusplus[i]=[j for j in range(1,len(slopes[i])) if (slopes[i][j]>0)&(slopes[i][j-1]<0)]

num_mp=[len(minusplus[i]) for i in range(len(numpdc))] 


print num_pm[0]
"""



"""naive maxima and corresponding time values you can do it with 5 or 10 or something else, 1 or two largest"""

naivemaxes=[0]*len(numpdc)
nmax_times=[0]*len(numpdc)
maxinds=[0]*len(numpdc)
maxerr=[0]*len(numpdc)
for i in range(len(numpdc)):
	naivemaxes[i]=[corrpdc[i][j] for j in range (len(numpdc[i])) if corrpdc[i][j] in heapq.nlargest(1, corrpdc[i][max(j-10,0):min(j+10, len(numpdc[i])-1): 1])]
	nmax_times[i]=[numtime[i][j] for j in range (len(numpdc[i])) if corrpdc[i][j] in heapq.nlargest(1, corrpdc[i][max(j-10,0):min(j+10, len(numpdc[i])-1): 1])]
	maxinds[i]=[j for j in range (len(numpdc[i])) if corrpdc[i][j] in heapq.nlargest(1, corrpdc[i][max(j-10,0):min(j+10, len(numpdc[i])-1): 1])]
	maxerr[i]=[err[i][j] for j in maxinds[i]]
"""numbers of naive maxima"""

len_nmax=np.zeros(len(numpdc))

for i in range(len(numpdc)):
	len_nmax[i]=len(naivemaxes[i])


"""
print max(len_nmax), min(len_nmax), np.mean(len_nmax), 
"""
"""
plt.scatter(len_nmax, yvals, color='red')
plt.show()

str5=[i for i in range(len(len_nmax)) if len_nmax[i]==max(len_nmax)]
str6=[i for i in range(len(len_nmax)) if len_nmax[i]==min(len_nmax)]
print str5,str6

plt.xlim()
plt.scatter(numtime[130], corrpdc[130], color='blue')
plt.scatter(nmax_times[130], naivemaxes[130],color='red')
plt.show()
"""
"""get peak and peak height above minima"""
"""get minima"""
"""Auto-correlation function of one maximum to next-good clustering"""

autopdcmax=[0]*len(numpdc)
for i in range(len(numpdc)):
	autopdcmax[i]=[naivemaxes[i][j+1] for j in range(len(naivemaxes[i])-1)]

mautocovs=np.zeros(len(numpdc))
mautocorrcoef=np.zeros(len(numpdc))

for i in range(len(numpdc)):
	mautocorrcoef[i]=np.corrcoef(naivemaxes[i][:-1:], autopdcmax[i])[0][1]
	mautocovs[i]=np.cov(naivemaxes[i][:-1:],autopdcmax[i])[0][1]

"""peak to peak slopes"""
ptpslopes=np.zeros(len(numpdc))
ppslopes=[0]*len(numpdc)
for i in range(len(numpdc)):
	ppslopes[i]=[abs((naivemaxes[i][j+1]-naivemaxes[i][j])/(nmax_times[i][j+1]-nmax_times[i][j])) for j in range(len(naivemaxes[i])-1)]

for i in range(len(numpdc)):
	ptpslopes[i]=np.mean(ppslopes[i])
"""
plt.scatter(ptpslopes, yvals)
plt.show()
"""
"""rms, mad, average deviation, """

"""Variation coefficient of time difference between successive maxima- periodicity?"""

maxdiff=[0]*(len(numpdc))
for i in range(len(numpdc)):
	maxdiff[i]=[nmax_times[i][j+1]-nmax_times[i][j] for j in range(len(naivemaxes[i])-1)]

periodicity=np.zeros(len(numpdc))
periodicityr=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	periodicity[i]=np.std(maxdiff[i])/np.mean(maxdiff[i])
	periodicityr[i]=np.sum(abs(maxdiff[i]-np.mean(maxdiff[i])))/np.mean(maxdiff[i])
naiveperiod=np.zeros(len(numpdc))

for i in range(len(numpdc)):
	naiveperiod[i]=np.mean(maxdiff[i])

"""variation coefficient of the maxima"""
maxvars=np.zeros(len(numpdc))
maxvarsr=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	maxvars[i]=np.std(naivemaxes[i])/np.mean(naivemaxes[i])

for i in range(len(numpdc)):
	maxvars[i]=np.std(naivemaxes[i])/np.mean(naivemaxes[i])
	maxvarsr[i]=np.sum(abs(naivemaxes[i]-np.mean(naivemaxes[i])))/np.mean(naivemaxes[i])
"""lombscargle of maxima
freqs=np.linspace(0.001,1,1000)
pmax=[0]*len(numpdc)
for i in range(len(numpdc)):
	pmax[i]=astroML.time_series.lomb_scargle(nmax_times[i],naivemaxes[i],maxerr[i],freqs)
	pmax[i]=astroML_addons.periodogram.lomb_scargle(nmax_times[i],naivemaxes[i],maxerr[i],freqs[i])

ampp=np.zeros(len(numpdc))
pfreq=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	ampp[i]=np.nanmax(pmax[i])
	pfreq[i]=len([freqs[j] for j in range(len(freqs)) if pmax[i][j]==ampp[i]])
print pfreq
print ampp
"""
"""
print 'This is it'
pl.scatter(pfreq, ampp)
pl.show()

print periodicity[92], np.mean(periodicity)
print maxvarsr[92], np.mean(maxvarsr)
"""

"""
pl.scatter(maxvarsr,ampp)
pl.show()
merge=np.vstack((num_zcross, maxvarsr))
merge=merge.T
kmeans = KMeans(n_clusters=5, random_state=rng).fit(merge)
result=np.round(kmeans.cluster_centers_, decimals=2)
result=result.T

plot_2D(merge,kmeans.labels_, ["c1","c2","c3","c4","c5"])
plt.scatter(result[0],result[1],color='red')
plt.show()"""
"""look at gini later"""


"""merge=np.vstack((mautocorrcoef, absmeansecder))
merge=merge.T
kmeans = KMeans(n_clusters=4, random_state=rng).fit(merge)
result=np.round(kmeans.cluster_centers_, decimals=2)
result=result.T
plt.scatter(result[0],result[1],color='orange')
plot_2D(merge,kmeans.labels_, ["c1","c2","c3","c4"])"""

"""naive minima and corresponding time values you can do it with 5 or 10 or something else"""

naivemins=[0]*len(numpdc)
nmin_times=[0]*len(numpdc)
mininds=[0]*len(numpdc)

for i in range(len(numpdc)):
	naivemins[i]=[corrpdc[i][j] for j in range (len(numpdc[i])) if corrpdc[i][j] in heapq.nsmallest(1, corrpdc[i][max(j-10,0):min(j+10, len(numpdc[i])-1): 1])]
	nmin_times[i]=[numtime[i][j] for j in range (len(numpdc[i])) if corrpdc[i][j] in heapq.nsmallest(1, corrpdc[i][max(j-10,0):min(j+10, len(numpdc[i])-1): 1])]
	mininds[i]=[j for j in range (len(numpdc[i])) if corrpdc[i][j] in heapq.nsmallest(1, corrpdc[i][max(j-10,0):min(j+10, len(numpdc[i])-1): 1])]

"""numbers of naive minima"""

len_nmin=np.zeros(len(numpdc))

for i in range(len(numpdc)):
	len_nmin[i]=len(naivemins[i])
"""ratio of successive minima-clustering would be great here,try maxima also"""
omin=[0]*len(numpdc)
emin=[0]*len(numpdc)
meanomin=np.zeros(len(numpdc))
meanemin=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	emin[i]=[naivemins[i][j] for j in range(len(naivemins[i])) if j%2==0]
	omin[i]=[naivemins[i][j] for j in range(len(naivemins[i])) if j%2!=0]
"""local secder dip"""
for i in range(len(numpdc)):
	meanemin[i]=np.mean(emin[i])
	meanomin[i]=np.mean(omin[i])
"""plt.scatter(meanomin, meanemin)"""
oeratio=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	oeratio[i]=meanomin[i]/meanemin[i]
"""
plt.scatter(abs(np.log(oeratio)),yvals)
plt.show()
"""
print 'check 2.5'

"""peak height and width

heights=[]
widths=[]
meanheight=np.zeros(len(numpdc))
meanwidth=np.zeros(len(numpdc))

for i in range(len(numpdc)):
	heights.append([])
	widths.append([])
	for j in range(len(naivemaxes[i])-1):
		k=maxinds[i][j]
		while((slopes[i][k]>abs(0.05*np.std(slopes[i]))) & (k<len(slopes[i]))):
			k-=1
		l=maxinds[i][j]
		while((slopes[i][l]<abs(0.05*np.std(slopes[i]))) & (k<len(slopes[i]))):
			l+=1
		widths[i].append(max(-numtime[i][k]+numtime[i][maxinds[i][j]],-numtime[i][maxinds[i][j]]+numtime[i][l]))	
		heights[i].append(max(abs(corrpdc[i][k]-corrpdc[i][maxinds[i][j]]),abs(corrpdc[i][maxinds[i][j]]-corrpdc[i][l])))

for i in range(len(numpdc)):
	meanheight[i]=np.mean(heights[i])
	meanwidth[i]=np.mean(widths[i])
"""
"""
asymm=[0]*len(numpdc)
pasymm=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	asymm[i]=[np.mean(slopes[i][max(j-int((meanwidth[i]*len(numpdc[i])/(max(numtimes[i])-min(numtimes[i]))),0):j-1:1])/np.mean(slopes[i][j:min(j+int((meanwidth[i]*len(numpdc[i])/(max(numtimes[i])-min(numtimes[i]))),len(slopes[i])):1]) for j in maxinds[i]] 
	slopes[i][max(j-int(meanwidth[i]),0):j-1:1]
	print asymm[i]
for i in range(len(numpdc)):
	if asymm[i]!=[]:
		print 'yes'
		pasymm[i]=np.nansum(asymm[i])/len(asymm[i])
print pasymm

plt.scatter(pasymm,yvals)
"""

"""peak-wise asymmetry- might backfire for high numpm cases if you use 6
meantimediff=[0]*len(numpdc)
for i in range(len(numpdc)):
	meantimediff[i]=np.mean([(numtime[i][j]-numtime[i][j-1]) for j in range(1, len(numpdc[i]))])

print meantimediff[0]

asymm=[0]*len(numpdc)
pasymm=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	asymm[i]=[np.mean(slopes[i][max(j-int(meanwidth[i]/(meantimediff[i])),0):j-1:1])/np.mean(slopes[i][j:min(j+int(meanwidth[i]/(meantimediff[i])),len(slopes[i])):1]) for j in maxinds[i]] 
	slopes[i][max(j-int(meanwidth[i]),0):j-1:1]

for i in range(len(numpdc)):
	if asymm[i]!=[]:
		pasymm[i]=np.nansum(asymm[i])/len(asymm[i])
print pasymm[0]
"""









"""merge=np.vstack((meanheight, meanwidth))
merge=merge.T
kmeans = KMeans(n_clusters=5, random_state=rng).fit(merge)
result=np.round(kmeans.cluster_centers_, decimals=2)
result=result.T

plt.figure()
plt.xlabel('varabsslope')
plt.ylabel('g_asymm')
plot_2D(merge,kmeans.labels_, ["c1","c2","c3","c4","c5"])
plt.scatter(result[0],result[1],color='green')
plt.show()"""

print 'check speed 3'

"""-----------------------------------------------------------------------------------"""

"""-------------------------AMPLITUDES and  PERCENTILES----------------------------------"""



"""secder near minima for transits"""
"""look at min and maxindices relationship, nmins vs nmaxes, broadness of periodogram peak, etc., hor line in corr pdc where most no of point densities"""

"""Amplitudes- you can try topten and percentile both"""
naive_amp=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	naive_amp=max(numpdc[i])-min(numpdc[i])

naive_amp_2=np.zeros(len(numpdc))
 
for i in range(len(numpdc)):
	naive_amp_2[i]=max(corrpdc[i])-min(corrpdc[i]) 


amp_2=np.zeros(len(numpdc))
amp =np.zeros(len(numpdc))

for i in range(len(numpdc)):
	amp[i]=np.percentile(numpdc[i],99)-np.percentile(numpdc[i],1)
	amp_2[i]=np.percentile(corrpdc[i],99)-np.percentile(corrpdc[i],1)

normnaiveamp=np.zeros(len(numpdc))
normamp=np.zeros(len(numpdc))

for i in range(len(numpdc)):
	normnaiveamp[i]=naive_amp_2[i]/np.mean(numpdc[i])
	normamp[i]=amp_2[i]/np.mean(numpdc[i])

"""median buffer percentile"""
mbp=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	mbp[i]=len([numpdc[i][j] for j in range(len(numpdc[i])) if (numpdc[i][j]<(medians[i]+0.1*amp_2[i])) & (numpdc[i][j]>(medians[i]-0.1*amp_2[i]))])/len(numpdc[i])


print 'naiveamp,amp2'

"""
TV
for i in range(len(numpdc)):
	print coeffvar[i]
	plt.scatter(numtime[i],corrpdc[i],color='blue')
	plt.scatter(numtime[i],numpdc[i],color='red')
	plt.show()
"""
"""
merge=np.vstack((naiveperiod, amp_2))
merge=merge.T
kmeans = KMeans(n_clusters=4, random_state=rng).fit(merge)
result=np.round(kmeans.cluster_centers_, decimals=2)
result=result.T
plt.scatter(result[0],result[1],color='orange')
plot_2D(merge,kmeans.labels_, ["c1","c2","c3","c4"])
"""


"""amp[i]=np.mean(heapq.nlargest(10, numpdc[i]))-np.mean(heapq.nsmallest(10, numpdc[i]))
amp_2[i]=np.mean(heapq.nlargest(10, corrpdc[i]))-np.mean(heapq.nsmallest(10, corrpdc[i])) 
gdors koi parameters, """

"""Percentiles"""
f595=np.zeros(len(numpdc))
f1090=np.zeros(len(numpdc))
f1782=np.zeros(len(numpdc))
f2575=np.zeros(len(numpdc))
f3267=np.zeros(len(numpdc))
f4060=np.zeros(len(numpdc))
mid20=np.zeros(len(numpdc))
mid35=np.zeros(len(numpdc))
mid50=np.zeros(len(numpdc))
mid65=np.zeros(len(numpdc))
mid80=np.zeros(len(numpdc))

for i in range(len(numpdc)):
	f595[i]=np.percentile(numpdc[i],95)-np.percentile(numpdc[i],5)
	f1090[i]=np.percentile(numpdc[i],90)-np.percentile(numpdc[i],10)	
	f1782[i]=np.percentile(numpdc[i], 82)-np.percentile(numpdc[i], 17)
	f2575[i]=np.percentile(numpdc[i], 75)-np.percentile(numpdc[i], 25)
	f3267[i]=np.percentile(numpdc[i], 67)-np.percentile(numpdc[i], 32)
	f4060[i]=np.percentile(numpdc[i], 60)-np.percentile(numpdc[i], 40)
	mid20[i]=f4060[i]/f595[i]
	mid35[i]=f3267[i]/f595[i]
	mid50[i]=f2575[i]/f595[i]
	mid65[i]=f1782[i]/f595[i]
	mid80[i]=f1090[i]/f595[i]

"""Percentamp"""
percentamp=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	percentamp[i]=max([(corrpdc[i][j]-medians[i])/medians[i] for j in range(len(corrpdc[i]))])
"""magratio"""

magratio=[(max(numpdc[i])-medians[i])/amp[i] for i in range(len(numpdc))]

"""---------------------------------------------------------------------"""



"""------------------------------LOMB SCARGLE OUTPOST---------------------------------
freqs=[np.linspace(0.001,1,1000)]*len(numpdc)
p=[0]*len(numpdc)

for i in range(len(numpdc)):
	p[i]=astroML_addons.periodogram.lomb_scargle(numtime[i],numpdc[i],numerr[i],freqs[i])

amp10=np.zeros(len(numpdc))
amp11=np.zeros(len(numpdc))
amp12=np.zeros(len(numpdc))
amp13=np.zeros(len(numpdc))
f1=np.zeros(len(numpdc))
f2=np.zeros(len(numpdc))
f3=np.zeros(len(numpdc))
ind=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	amp10[i]=max(p[i])
	f1[i]=[freqs[i][j] for j in range(len(freqs[i])) if p[i][j]==max(p[i])][0]
	ind[i]=[j for j in range(len(freqs[i])) if p[i][j]==max(p[i])][0]
	if 2*f1[i] in freqs[i]:
		amp11[i]=[p[i][j] for j in range(len(freqs[i])) if freqs[i][j]==2*f1[i]][0]
		ind[i]=[j for j in range(len(freqs[i])) if freqs[i][j]==2*f1[i]][0]
	else:
		amp11[i]=-0.05
	if 3*f1[i] in freqs[i]:
		amp12[i]= [p[i][j] for j in range(len(freqs[i])-1) if freqs[i][j]==3*f1[i]][0]
		ind[i]=[j for j in range(len(freqs[i])) if freqs[i][j]==3*f1[i]][0]
	else:
		amp12[i]=-0.05
	if 4*f1[i] in freqs[i]:
		amp13[i]= [p[i][j] for j in range(len(freqs[i])-1) if freqs[i][j]==4*f1[i]][0]	
		ind[i]=[j for j in range(len(freqs[i])) if freqs[i][j]==4*f1[i]][0]
	else:
		amp13[i]=-0.05

f1harmonicsratio=np.zeros(len(numpdc))

for i in range(len(numpdc)):
	f1harmonicsratio[i]=amp10[i]/amp11[i]
"""	
"""plt.scatter(f1,f1harmonicsratio)"""

"""amp10=np.zeros(len(numpdc))
amp11=np.zeros(len(numpdc))
amp12=np.zeros(len(numpdc))
amp13=np.zeros(len(numpdc))
amp2=np.zeros(len(numpdc))
amp3=np.zeros(len(numpdc))
f1=np.zeros(len(numpdc))
f2=np.zeros(len(numpdc))
f3=np.zeros(len(numpdc))
t1=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	amp10[i]=max(p[i])
	amp2[i]=heapq.nlargest(2,p[i])[1]	
	amp3[i]=heapq.nlargest(3,p[i])[2]
	f1[i]=[freqs[j] for j in range(len(freqs)) if p[i][j]==max(p[i])][0]	    	
	f2[i]=[j for j in range(len(freqs)) if p[i][j]==max(p[i])][0]
	f3[i]=[j for j in range(len(freqs)) if p[i][j]==max(p[i])][0]"""
	
"""numbers of naive maxima

ls_nmax=np.zeros(len(numpdc))

for i in range(len(numpdc)):
	ls_nmax[i]=len(naivemaxes[i])
"""
"""mean peak height

lsmeanpeakheight=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	lsmeanpeakheight[i]=np.mean(naivemaxes[i])
"""
"""flux-power integral or varibility measure they say
flux=[sum(p[i]) for i in range (len(numpdc))]"""
"""plt.scatter(np.log(flux),yvals)
plt.show()"""

"""-----------------------------------------------------------------"""

print 'check speed 4'


"""Autocorrs and sautocorrs-biased by data going both ways drastically"""

autopdc=[0]*len(numpdc)
for i in range(len(numpdc)):
	autopdc[i]=[numpdc[i][j+1] for j in range(len(numpdc[i])-1)]

autocovs=np.zeros(len(numpdc))
autocorrcoef=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	autocorrcoef[i]=np.corrcoef(numpdc[i][:-1:], autopdc[i])[0][1]
	autocovs[i]=np.cov(numpdc[i][:-1:],autopdc[i])[0][1]
 
sautopdc=[0]*len(slopes)
for i in range(len(slopes)):
	sautopdc[i]=[slopes[i][j+1] for j in range(len(slopes[i])-1)]

sautocovs=np.zeros(len(slopes))
sautocorrcoef=np.zeros(len(slopes))
for i in range(len(slopes)):
	sautocorrcoef[i]=np.corrcoef(slopes[i][:-1:], sautopdc[i])[0][1]
	sautocovs[i]=np.cov(slopes[i][:-1:],sautopdc[i])[0][1]




"""Peak flatness-log plot very similar to random not anympre,im putting meanwidth instead of 6 now"""

flatness=[0]*len(numpdc)

for i in range(len(numpdc)):
	flatness[i]=[np.mean(corrslopes[i][max(0,j-6):min(j-1, len(corrslopes[i])-1):1])- np.mean(corrslopes[i][max(0,j):min(j+4, len(corrslopes[i])-1):1]) for j in range(len(corrslopes[i])) if corrpdc[i][j] in naivemaxes[i]]


flatmean=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	flatmean[i]=np.nansum(flatness[i])/len(flatness[i])




"""Trough flatness"""
tflatness=[0]*len(numpdc)

for i in range(len(numpdc)):
	tflatness[i]=[-np.mean(corrslopes[i][max(0,j-6):min(j-1, len(corrslopes[i])-1):1])+ np.mean(corrslopes[i][max(0,j):min(j+4, len(corrslopes[i])-1):1]) for j in range(len(corrslopes[i])) if corrpdc[i][j] in naivemins[i]]
	
tflatmean=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	tflatmean[i]=np.nansum(tflatness[i])/len(tflatness[i])
	
	

"""troundness coming off well, but no real measure
of 'linearity' for var in slopes, as many are convex
at the bottom, mean slope vs flatness at trough for transits?"""

"""Peak roundness r1"""

roundness=[0]*len(numpdc)

for i in range(len(numpdc)):
	roundness[i]=[np.mean(secder[i][max(0,j-6):min(j-1, len(secder[i])-1):1]) + np.mean(secder[i][max(0,j+1):min(j+6, len(secder[i])-1):1]) for j in range(len(secder[i])) if corrpdc[i][j+1] in naivemaxes[i]]

roundmean=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	roundmean[i]=np.nansum(roundness[i])/len(roundness[i])




"""trough roundness-couple with flatness, alienate transits"""

troundness=[0]*len(numpdc)

for i in range(len(numpdc)):
	troundness[i]=[np.mean(secder[i][max(0,j-6):min(j-1, len(secder[i])-1):1]) + np.mean(secder[i][max(0,j+1):min(j+6, len(secder[i])-1):1]) for j in range(len(secder[i])) if corrpdc[i][j+1] in naivemins[i]]
	
troundmean=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	troundmean[i]=np.nansum(troundness[i])/len(troundness[i])
	"""print troundmean[i]"""
	"""print tflatmean[i]"""
	"""print num_zcross[i] mean height and width,maxslope,skewslope
	print 'height', meanheight[i],'width', meanwidth[i]
	plt.xlim(970,975)"""
	"""plt.scatter(numtime[i],corrpdc[i],color='blue')
	plt.scatter(nmin_times[i], naivemins[i],color='green')
	plt.scatter(nmax_times[i],naivemaxes[i],color='red')
	plt.show()
	plt.scatter(numtime[i],corrpdc[i])
	plt.scatter(nmin_times[i], naivemins[i],color='green')
	plt.scatter(nmax_times[i],naivemaxes[i],color='red')
	plt.show()"""

roundrat=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	roundrat[i]=roundmean[i]/troundmean[i]

flatrat=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	flatrat[i]=flatmean[i]/tflatmean[i]


"""pl.scatter(np.log(abs(roundmean))*abs(roundmean)/roundmean, np.log(abs(flatrat))*abs(flatrat)/flatrat)
pl.show()"""
"""
scatter flatnesses separately, and troundmean
pl.scatter(g_asymm,skewslope)
pl.show()
"""

"""---------------------------PCA and Clustering Algorithms---------------------------------
nperiodicity=periodicity/max(periodicity)-min(periodicity)

x=np.vstack((periodicity,mautocorrcoef,g_asymm, autocorrcoef,np.log(abs(roundmean))*abs(roundmean)/roundmean,f1,mid20,skewslope,np.log(num_zcross),flatrat))
x2=np.vstack((g_asymm,skewslope))
xnorm=((periodicity/))
xtr=x.T
"""
"""try meanwidth, pasymm, skewslopes, absmeansecder.numnspikes too long(?), logflux, amp10,amp_2, varabsslope-log and otherwise,g_asymm,maxslope, coeffvar,numoutliers,num_zcross,roundmean, f1 try kurt skews, numpm,logamp2, depth*tflatness, pasymm, roundrat and its log, periodogram of maxes"""

"""PCA
pca = PCA(n_components=2, whiten=True).fit(xtr)
print pca.components_ 
print pca.explained_variance_ratio_ 
x_pca = pca.transform(xtr)

DBSCAN
db = DBSCAN(eps=5, min_samples=3,metric='euclidean').fit(xtr)
core_samples = db.core_sample_indices_
labels = db.labels_
components=db.components_
n_clusters_ = len(set(labels)) 

print labels

print n_clusters_
"""
"""
print core_samples
print components
"""
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

for i in range(n_clusters_):
	print '--------------'
	

"""
"""KMEANS
kmeans = KMeans(n_clusters=4, random_state=rng).fit(xtr)
result=np.round(kmeans.cluster_centers_, decimals=2)
result=result.T
print result

plot_2D(x_pca,kmeans.labels_, ["c1","c2","c3","c4"])
"""
"""WARD

ward=Ward(n_clusters=4, connectivity=None, copy=True, n_components=None, compute_full_tree='auto').fit(x_pca)
print ward.labels_
print ward.n_leaves_
print ward.children_
print ward.n_components_

plot_2D(x_pca,ward.labels_,["c1","c2","c3","c4"])

"""
"""Writing in Parameters- conc is transpose of np.vstack of all previous ones"""
"""
conc=np.vstack((meanmedrat, skews, varss, coeffvar, stds, numoutliers, numnegoutliers, kurt, mad, maxslope, minslope, meanpslope, meannslope, g_asymm, rough_g_asymm, diff_asymm, skewslope, varabsslope, varslope, meanabsslope, absmeansecder, num_pspikes, num_nspikes, num_sdspikes, num_sdspikes2, num_zcross, num_pm, len_nmax, mautocorrcoef, ptpslopes, periodicity, periodicityr, naiveperiod, maxvars, maxvarsr, oeratio, amp_2, normamp, mid20, mid35, mid50, mid65, mid80, sautocorrcoef, autocorrcoef, flatmean, tflatmean, roundmean, troundmean, roundrat, flatrat))


conc=conc.T

print len(longtermtrend),len(numout1s),len(stdratio),len(pstrend),len(mbp), len(percentamp), len(magratio), len(posoutliers)
"""
conc=np.vstack((longtermtrend, meanmedrat, skews, varss, coeffvar, stds, numoutliers, numnegoutliers, numposoutliers, numout1s, kurt, mad, maxslope, minslope, meanpslope, meannslope, g_asymm, rough_g_asymm, diff_asymm, skewslope, varabsslope, varslope, meanabsslope, absmeansecder, num_pspikes, num_nspikes, num_sdspikes, num_sdspikes2,stdratio, pstrend, num_zcross, num_pm, len_nmax, len_nmin, mautocorrcoef, ptpslopes, periodicity, periodicityr, naiveperiod, maxvars, maxvarsr, oeratio, amp_2, normamp,mbp, mid20, mid35, mid50, mid65, mid80, percentamp, magratio, sautocorrcoef, autocorrcoef, flatmean, tflatmean, roundmean, troundmean, roundrat, flatrat))


conc=conc.T

"""
textfile = file('new_file_2.txt','wt')

print len(xtr)
print len(xtr[0])
"""
f=open("new_file_2.txt","a")
for i in range(len(conc)):
	for j in range(len(conc[i])):
		f.write(str(conc[i][j])+' ')
	f.write('\n')
f.close()





"""--------------------------------------------------------------------------------------------"""

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
for i in range(len(numpdc)):
	if ward.labels_[i]==1:
	
		plt.scatter(numtime[i],numpdc[i],color='red')
		plt.show()
		plt.xlim(970,975)
		plt.scatter(numtime[i],numpdc[i],color='red')
		plt.show()
print 'three' 
for i in range(len(numpdc)):	
	if ward.labels_[i]==2:
		print 'three' 
		plt.scatter(numtime[i],numpdc[i],color='red')
		plt.show()
		plt.xlim(970,975)
		plt.scatter(numtime[i],numpdc[i],color='red')
		plt.show()
print 'four' 
for i in range(len(numpdc)):
	if ward.labels_[i]==3:
		print 'four'
		plt.scatter(numtime[i],numpdc[i],color='red')
		plt.show()
		plt.xlim(970,975)
		plt.scatter(numtime[i],numpdc[i],color='red')
		plt.show()		 

"""




"""remember xlim can be from 970 to 975 or 972 to 973 to suit your purpose"""

"""
merge=np.vstack((roundmean,troundmean))
merge=merge.T
kmeans = KMeans(n_clusters=5, random_state=rng).fit(merge)
result=np.round(kmeans.cluster_centers_, decimals=2)
result=result.T

plot_2D(merge,kmeans.labels_, ["c1","c2","c3","c4","c5"])
plt.scatter(result[0],result[1],color='red')
plt.show()
"""
"""
print 'rounded'

for i in range(len(numpdc)):
	if roundmean[i]<-0.05:
		print roundmean[i],troundmean[i]
		plt.xlim(970,975)
		plt.scatter(numtime[i],corrpdc[i],color='blue')
		plt.scatter(numtime[i],numpdc[i],color='red')
		plt.show()
"""
"""
print 'rounded'

for i in range(len(numpdc)):
	if troundmean[i]<-0.25:
		print roundmean[i],troundmean[i]
		plt.xlim(970,975)
		plt.scatter(numtime[i],corrpdc[i],color='blue')
		plt.scatter(numtime[i],numpdc[i],color='red')
		plt.show()
"""

print 'check speed 5'

"""
DBSCAN NOT WORKING
merge=np.vstack((amp_2, naiveperiod))
merge=merge.T
dbscan = DBSCAN(metric='euclidean').fit(merge)
print dbscan.labels_

result=np.round(kmeans.cluster_centers_, decimals=2)
result=result.T
plot_2D(merge,dbscan.labels_, ["c1","c2","c3","c4","c5"])
plt.scatter(result[0],result[1],color='red')
plt.show()"""

"""r2-this seems less refined but another purpose
varslopepeak=[0]*len(numpdc)
for i in range(len(numpdc)):
	varslopepeak[i]=[np.var(absslopes[i][max(0,j-5):min(j+5, len(absslopes[i])-1):1]) for j in range(len(absslopes[i])) if corrpdc[i][j] in naivemaxes[i]]

  
meanvsp=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	meanvsp[i]=np.mean(varslopepeak[i])
"""



"""try peak asymm, identify maxima, go halfway down the period, see slope asymmetry,"""






"""plt.scatter(len_nmax, len_nmin)
plt.show()"""

"""a note- see corrcoef between various parameters and significance against null hypothesis and no clustering"""



"""
print np.argmin(roundness)

plt.xlim(980,982)
plt.scatter(numtime[np.argmin(roundness)], numpdc[np.argmin(roundness)])
plt.show()

plt.xlim(945,946)
plt.scatter(numtime[np.argmax(roundness)], numpdc[np.argmax(roundness)])
plt.show()



plt.scatter(mautocorrcoef, autocorrcoef)
plt.show()

plt.scatter(naiveperiod, amp_2)
plt.show()



plt.scatter(absmeansecder, numoutliers)
plt.show()


plt.scatter(coeffvar, varss)
plt.show()


plt.scatter(g_asymm, num_zcross)
plt.show()

plt.scatter(len_nmax, periodicity)
plt.show()

plt.scatter(meanmedrat, skews)
plt.show()

plt.scatter(numtime[np.argmin(meanvsp)], numpdc[np.argmin(meanvsp)])

print np.argmax(meanvsp)==np.argmax(flatness) or np.argmax(meanvsp)==np.argmin(flatness)

plt.show()

"""
"""
print meanvsp[0], meanvsp[1]
plt.xlim(950,980)
plt.scatter(numtime[0],numpdc[0])
plt.show()

plt.scatter(numtime[1], numpdc[1])
plt.show()"""

"""periodogram-64t problem

freqs = np.linspace(0.01, 1, 1000)

arrt=np.array(numtime[0])
arrpd=np.array(numpdc[0])
arrt.astype('float64')
arrt.astype('float64')
periodogram = spectral.lombscargle(arrt, arrpd, freqs)


plt.scatter(freqs,periodogram,color='red')
plt.show()
"""
"""roundness of peak-1-secder robust way, though weighted would be better

roundness=[0]*len(numpdc)

for i in range(len(numpdc)):
	roundness[i]=[np.mean(secder[i][max(0,j-6):min(j-1, len(secder[i])-1):1]) + np.mean(secder[i][max(0,j+1):min(j+6, len(secder[i])-1):1]) for j in range(len(secder[i])) if corrpdc[i][j] in naivemaxes[i]]


roundmean=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	roundmean[i]=np.mean(roundness[i])
print np.mean(roundness[0])
"""
"""roundness of peak-2-var way

varslopepeak=[0]*len(numpdc)
for i in range(len(numpdc)):
	varslopepeak[i]=[np.var(absslopes[i][max(0,j-5):min(j+5, len(absslopes[i])-1):1]) for j in range(len(absslopes[i])) if corrpdc[i][j] in naivemaxes[i]]

  
meanvsp=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	meanvsp[i]=np.mean(varslopepeak[i])
"""



"""
plt.scatter(numoutliers, mad)
plt.show()

plt.scatter(roundmean, num_zcross)

plt.show()
"""




"""
peakround=np.zeros(len(numpdc))

for i in range(len(numpdc)):
	
roundness of peak-2
yet a problem with this code- invalid numpy operation-invalid double scalar
varslopepeak=[0]*len(numpdc)
for i in range(len(numpdc)):
	varslopepeak[i]=[np.var(absslopes[i][max(0,j-5):min(j+5, len(absslopes[i])-1):1]) for j in range(len(absslopes[i])) if numpdc[i][j] in naivemaxes[i]]  

meanvsp=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	meanvsp[i]=np.mean(varslopepeak[i])


plt.scatter(meanvsp, yvals)
plt.show()

plt.xlim(950, 954)
plt.scatter(numtime[47], numpdc[47])
plt.show()



str9=[i for i in range(len(ratio)) if ratio[i]==max(ratio)]
str10=[i for i in range(len(ratio)) if ratio[i]==min(ratio)]
print str9,str10
175, 47


str7=[i for i in range(len(periodicity)) if periodicity[i]==max(periodicity)]
str8=[i for i in range(len(periodicity)) if periodicity[i]==min(periodicity)]
print str7,str8
8, 130 (i think)

plt.scatter(ratio, yvals)
plt.show()
nice clustering for this one

plt.xlim(0,0.1)
plt.ylim(0,1000)

plt.scatter(periodicity, len_nmax)

corrcoef also between this is 0.23
in fact 0.3 after taking scatter per mean

plt.show()

strong anticorrelation between these; -0.23 or -0.3 

I hear len_max or period is related to density-inversely
proportional as 1/sqrt(rho) 
giants then have less periodic as opposed to white dwarfs? 

"""
"""variance of maxima themselves, skew etc."""
"""Difference between successive maxima"""
"""
roundness/flatness of peak, peakwise asymmetry and otherwise-meanpslope versus meannslope 


roundness/vness of peak measured by weighted mean second der around maximum, 
less this is, more the 'v' actually count the minus ones to the left plus ones to right,
but also scatter of slopes around peaks is a good way
asymmetry measured by difference in pslope and nslope for corrpdc, also their absolute
means, similar parameter for secder might cluster out transits; there maxima slopes
may not vary steeply, but KOI ones will; look for 'outliers' beyond say 5 sigma




TROUBLE STARTS HERE

naive_maxes=[]
for i in range(len(numpdc)):
	naive_maxes.append([])
	for j in range(len(numpdc[i])):		
		if((np.mean(slopes[i][max(j-10,0):min(j, len(numpdc[i])-1): 1])>0) & (np.mean(slopes[i][max(j,0):min(j+10, len(numpdc[i])-1):1])<0)):
			naive_maxes[i].append(numpdc[i][j])

ERROR:double scalar

naive_maxes=[0]*len(numpdc[i])
for i in range(len(numpdc)):
	for j in range(len(numpdc[i])):		
		naive_maxes[i]=[numpdc[i][j] for j in range (len(numpdc[i])) if (np.mean(slopes[i][max(j-10,0):min(j, len(numpdc[i])-1): 1])>abs(meanslope[i])) & (np.mean(slopes[i][max(j,0):min(j+10, len(numpdc[i])-1):1])<-0.1)]
			





maxes=[]
for i in range(len(numpdc)):
	maxes.append([])
	for j in range(len(numpdc[i])):	
		if numpdc[i][j] in heapq.nlargest(5, numpdc[i][max(i-10,0):min(i+10, len(numpdc[i])-1): 1]):
			if((np.mean(slopes[i][max(j-10,0):min(j, len(numpdc[i])-1): 1])>0) & (np.mean(slopes[i][max(j,0):min(j+10, len(numpdc[i])-1):1])<0)):
				maxes[i].append(numpdc[i][j])

TROUBLE ENDS HERE




longtermtrend=np.zeros(len(numpdc))

for i in range(len(numpdc)):
     longtermtrend[i]=np.polyfit(numtime[i], numpdc[i], 1)[0]

meanltt=np.mean(longtermtrend)
medianltt=np.median(longtermtrend)

y_offset=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	y_offset[i]=np.polyfit(numtime[i], numpdc[i], 1)[1]




pdcdiff=[0]*len(numpdc)
 
for i in range(len(numpdc)):
	pdcdiff[i]=[(numpdc[i][j+1]-numpdc[i][j]) for j in range(len(numpdc)-1)]

meanpdcdiff=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	meanpdcdiff[i]=np.mean(pdcdiff[i]) 




linpoints=[0]*len(numtime)
for i in range(len(numtime)):
	linpoints[i]=[val for val in range(len(secder[i])) if val<=meansecder[i]/100]

num_lin=np.zeros(len(numtime))
for i in range(len(numpdc)):
	num_lin[i]=len(linpoints[i])

maxslope_naive=np.zeros(len(numpdc))
maxslope=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	maxslope_naive[i]=max(slopes[i])
	maxslope[i]=np.percentile(slopes[i],95)-np.percentile(slopes[i],5)
	
similarly flatpoints



TROUBLE STARTS HERE

hist=[]



for i in range(len(numpdc)):
	hist.append([])
	for j in range(len(numpdc[i])):
		hist[i].append([])
		for k in range(len(numtime[i])):
 			if (numpdc[i][k]>(numpdc[i][j]-meanpdcdiff[i]))&(numpdc[i][k]<(numpdc[i][j]+meanpdcdiff[i])):
				hist[i][j].append(numtime[i][k]) 



hist=[]
pdcdiff=[0]*185
for i in range(len(numpdc)):
	pdcdiff[i]=[(numpdc[i][j+1]-numpdc[i][j]) for j in range(len(numpdc)-1)]

meanpdcdiff=np.zeros(len(numpdc))
for i in range(len(numpdc)):
	meanpdcdiff[i]=np.mean(pdcdiff[i]) 

naive_hist=[]

for i in range(len(numpdc)):
	naive_hist.append([])
	for j in range(len(numpdc[i])):
		naive_hist[i].append(len([k for k in range(len(numtime[i])) if (numpdc[i][k]>(numpdc[i][j]-0.2)) & (numpdc[i][k]<(numpdc[i][j]+0.2))]))
				 
naive_hist[0]

arrt=np.array(numtime[0])
arrpd=np.array(numpdc[0])
arrt.astype('float64')
arrt.astype('float64')
periodogram = sp.signal.lombscargle(arrt, arrpd, freqs)






http://en.wikipedia.org/wiki/Statistical_dispersion- look at this for other 'dispersion measures.


"""
