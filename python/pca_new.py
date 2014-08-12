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

smallsample = np.load('../data/SmallSample.npy')
logsmallsample = np.load('../data/LogSmallSample.npy')
keplerids185 = np.load('../data/keplerids185.npy')

n=len(smallsample)
q = 60
testdata = np.empty([n,q])
'''
j=0
for i in range(0,60):
    mean = np.mean(logdata[:,i])
    std = np.mean(logdata[:,i])
    testdata[:,j] = (logdata[:,i]-mean)/std
    j=j+1
'''

j=0
for i in range(0,60):
    mean = np.mean(smallsample[:,i])
    std = np.mean(smallsample[:,i])
    testdata[:,j] = (smallsample[:,i]-mean)/std
    j=j+1

params = np.empty([len(testdata),59])
j=0
for i in range(0,60):
    if i not in [22]:
        params[:,j] = testdata[:,i]
        j=j+1
for i in range(0,len(params)):
    for j in range(0,len(params[i])):
        if np.isnan(params[i,j]) or np.isinf(params[i,j]): print i, j, params[i,j]
print smallsample.shape
print params.shape

'''
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

p = len(rrlyrae + instrip + detached + semidet + overcontact + ellipsoid)
kidlist = [rrlyrae, instrip, detached, semidet, overcontact, ellipsoid]
testlist = [rtest, itest, dtest, sdtest, octest, eltest]
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
        temp = params[np.where(smallsample[:,0]==j)[0][0]]
        temp2 = plotdata[np.where(smallsample[:,0]==j)[0][0]]
        for k in range(0,q):
            training[h,k] = temp[k]
        for k in range(0,q):
            pd[h,k] = temp2[k]
        h = h+1
'''
map = PCA(n_components=59)

#X = map.fit_transform(params)
#tempcomponents = map.components_
#tempratio = map.explained_variance_ratio_
#variance = np.empty([19,1])
#print params3
'''
j=0
params2 = np.empty([len(training),60])
for i in range(0,60):
    if i<=3 or 10<=i<=16 or i==20:
        params2[:,j] = training[:,i]
        j=j+1
'''    
X = map.fit_transform(params)

print map.explained_variance_ratio_
print map.explained_variance_ratio_.sum()

fig = pl.figure(1)
pl.clf()

l = pl.plot(np.zeros([59]),c='black')
l0 = pl.plot(map.components_[0],c='blue')
l1 = pl.plot(map.components_[1],c='green')
l2 = pl.plot(map.components_[2],c='orange')
l3 = pl.plot(map.components_[3],c='red')
pl.show()
