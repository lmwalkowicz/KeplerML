# reads in both both .sav files: the long-cadence data and the variable supplement
# removes all objects that lack data like effective temperatures
# writes the dataset.npy and logdata.npy files
# also building the training and test sets and runs a Gaussian Naive Bayes classifier

import numpy as np
import pyfits as pf
import itertools
import pickle
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

f = open('../data/SmallSample.txt','rb')

x = [row.strip().split(' ') for row in f]
x2 = [row.strip().split(' ') for row in f2]
smallsample = np.empty([len(x),len(x[0])])

for i in range(0,len(x)):
    for j in range(0,len(x[0])):
        print i,j,x[i][j]
        smallsample[i,j] = x[i][j]

logsmallsample = np.log10(smallsample)

print smallsample[0]

np.save('../data/SmallSample',smallsample)
np.save('../data/LogSmallSample',logsmallsample)

#rrlyrae = [5520878, 3733346, 5299596, 6070714, 6100702,   6763132, 6936115, 7176080]
#rtest = [7742534, 7988343, 8344381, 9508655, 9591503,   9947026, 10789273, 11802860]

#instrip = [2571868, 2987660, 3629496, 5356349, 5437206,   6668729, 7304385, 7974841, 8018827, 8324268]
#itest = [8351193, 8489712, 8915335, 9291618, 9351622,   10537907, 10974032, 11572666, 11874676, 12153021]

#detached = [1026032, 1026957, 1433962, 1571511, 1725193,   1996679, 2010607, 2162635, 2162994, 2305372,   2305543, 2306740]
#dtest = [2308957, 2309587, 2309719, 2437452, 2438070,   2440757, 2442084, 2445134, 2447893, 2556127,   2557430, 2576692]

#semidet = [4947528, 4949770, 5077994, 5120793, 5211385,   5215999, 5218441, 5374999, 5471619, 5774375,   5785586, 5792093, 5809827]
#sdtest = [5823121, 6283224, 6302051, 6353203, 6432059,   6606653, 6669809, 6692340, 6836140, 6852488,   6865626, 6962901, 7031714]

#overcontact = [7821450, 7830460, 7835348, 7839027, 7871200,   7877062, 7878402, 7879404, 7881722, 7889628,   7950962, 7973882, 7977261]
#octest = [8004839, 8035743, 8039225, 8053107, 8108785,   8111387, 8122124, 8143757, 8177958, 8190491,   8190613, 8192840, 8241252]

#ellipsoid = [9848190, 9898401, 9909497, 9948201, 10028352,   10030943, 10032392, 10123627, 10135584, 10148799,   10155563, 10285770, 10288502, 10291683, 10351735,   10417135]
#eltest = [10481912, 10600319, 10619506, 10855535, 11135978,   11336707, 11572643, 11714337, 11722816, 11751847,   11825204, 11875706, 12055421, 12059158, 12121738,   12166770]

#p = len(rrlyrae + instrip + detached + semidet + overcontact + ellipsoid + uncertain)
#kidlist = [rrlyrae, instrip, detached, semidet, overcontact, ellipsoid, uncertain]
#testlist = [rtest, itest, dtest, sdtest, octest, eltest, utest]
#col=np.empty([p],dtype='S10')
