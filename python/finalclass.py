# runs a random forest classifier
# prints the classification accuracy in each case
# NOTE: work in progress

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
from sklearn import svm
from astroML.classification import GMMBayes
#from __future__ import print_function

dataset = np.load('../data/SmallSample.npy')
logdata = np.load('../data/LogSmallSample.npy')
dataset2 = np.load('../data/dataset.npy')
logdata2 = np.load('../data/logdata.npy')

n2=len(dataset)
n22=len(dataset2)
print n2
q=14
q2=12
testdata = np.empty([n2,q])
plotdata = np.empty([n2,q])
testdata2 = np.empty([n22,q2])
plotdata2 = np.empty([n22,q2])

#order: teff,   logg,   maxpkht, sndht, maxper, sndper,
#       maxflx, sndflx, range,   rms4,  mdv3,   pdcvar

#50.0: null,5,7,12,13,14,20,21,23,24,26,29,34,35,42,46,52
#49.5: 4,43,51
#49.0: 2,16

#possibly omit 4,16,43
j=0
#for i in [25,56,43,52,30,29,7,53,4,16,37,34,44,40,10,9,51,58,6,2,26,31]:
for i in [2,4,7,16,23,26,34,37,43,46,51,52,53,56]:
#for i in [2,4,5,7,12,16,20,23,24,26,29,34,43,46,51]:
    mean = np.mean(dataset[:,i])
    std = np.mean(dataset[:,i])
    testdata[:,j] = (dataset[:,i]-mean)/std
    '''
    if i!=2 and i!=3:
        plotdata[:,j] = testdata[:,i]
    else:
        plotdata[:,j] = dataset[:,i]
    '''
    j=j+1

rms4 = testdata2[:,15]

j=0
for i in [2,3,15,18,16,19,17,20,5,24,7,28]:
    mean = np.mean(logdata2[:,i])
    std = np.mean(logdata2[:,i])
    testdata2[:,j] = (logdata2[:,i]-mean)/std
    j=j+1

teff = dataset[:,2]
logg = dataset[:,3]

rrlyrae = [32,40,44,51,52, 61,66,72]
rtest = [74,88,104,118,120, 134,162,175]
instrip = [27,30,31,41,42, 58,73,86,90,103]
itest = [106,107,108,111,114, 157,167,171,177,182]
detached = [0,1,2,3,4, 5,6,7,9,10, 11,12]
dtest = [13,14,15,17,18, 19,20,21,23,24, 26,28]
semidet = [33,34,35,36,37, 38,39,43,46,47, 48,49]
sdtest = [50,53,54,55,56, 57,59,60,63,64, 65,67]
overcontact = [76,77,78,79,80, 81,82,83,84,85, 87,89]
octest = [91,92,93,94,95, 96,97,98,99,100, 101,102]
ellipsoid = [130,132,133,136,138, 139,140,142,143,145, 146,149,150,151,153, 155]
eltest = [156,159,160,166,168, 169,170,172,173,174, 176,178,179,180,181, 183]
uncertain = [110,112,113,115,116, 117,119,121,122,123, 124,125]
utest = [126,128,129,131,137, 141,147,152,154,158, 161,165]

rrlyrae2 = [3733346, 5299596, 5520878, 6070714, 6100702,   6763132, 6936115, 7176080]
rtest2 = [7742534, 7988343, 8344381, 9508655, 9591503,   9947026, 10789273, 11802860]

instrip2 = [2571868, 2987660, 3629496, 5356349, 5437206,   6668729, 7304385, 7974841, 8018827, 8324268]
itest2 = [8351193, 8489712, 8915335, 9291618, 9351622,   10537907, 10974032, 11572666, 11874676, 12153021]

detached2 = [1026032, 1026957, 1433962, 1571511, 1725193,   1996679, 2010607, 2162635, 2162994, 2305372,   2305543, 2306740]
dtest2 = [2308957, 2309587, 2309719, 2437452, 2438070,   2440757, 2442084, 2445134, 2447893, 2556127,   2557430, 2576692]

semidet2 = [4947528, 4949770, 5077994, 5120793, 5211385,   5215999, 5218441, 5471619, 5774375, 5785586,   5792093, 5809827]
sdtest2 = [5823121, 6283224, 6302051, 6353203, 6432059,   6606653, 6669809, 6692340, 6836140, 6852488,   6865626, 6962901]

overcontact2 = [7830460, 7835348, 7839027, 7871200, 7877062,   7879404, 7881722, 7889628, 7950962, 7973882,   7977261, 8004839]
octest2 = [8035743, 8039225, 8053107, 8108785, 8111387,   8122124, 8143757, 8177958, 8190491, 8190613,   8192840, 8241252]

ellipsoid2 = [9848190, 9898401, 9909497, 9948201, 10028352,   10030943, 10032392, 10123627, 10135584, 10148799,   10155563, 10285770, 10288502, 10291683, 10351735,   10417135]
eltest2 = [10481912, 10600319, 10619506, 10855535, 11135978,   11336707, 11572643, 11714337, 11722816, 11751847,   11825204, 11875706, 12055421, 12059158, 12121738,   12166770]

uncertain2 = [9237533, 9347868, 9347955, 9456920, 9469350,   9480516, 9532591, 9596355, 9655187, 9713664,   9716456, 9724080]
utest2 = [9724220, 9832227, 9835416, 9874575, 9964422,   10086746, 10264744, 10350225, 10388897, 10556068,   10684673, 10799558]

quiet = [0]*100
qtest = [0]*100

qui=0
j=0
while 1:
    #if (logg[j]>=(6.0-0.0004*teff[j]) or logg[j]>=4.0) and logdata[j,5]<=1 and logdata[j,15]<=1 and teff[j]<=6100:
    if (logg[j]>=(6.0-0.0004*teff[j]) or logg[j]>=4.0) and rms4[j]<-1 and teff[j]<=6100:
        if qui<100: quiet[qui] = dataset[j,0]
        elif qui<200: qtest[qui-100] = dataset[j,0]
        qui = qui+1
    if qui>=200: break
    j = j+1


p2 = len(rrlyrae + instrip + detached + semidet + overcontact + ellipsoid)
p1 = p2 + 100
kidlist1 = [rrlyrae, instrip, detached, semidet, overcontact, ellipsoid]
testlist1 = [rtest, itest, dtest, sdtest, octest, eltest]
kidlist2 = [rrlyrae2, instrip2, detached2, semidet2, overcontact2, ellipsoid2]
testlist2 = [rtest2, itest2, dtest2, sdtest2, octest2, eltest2]

col=np.empty([2*p2,5],dtype='S10')
labels = np.empty([2*p2,5])

training = np.empty([p2,q])
sampletest = np.empty([p2,q])
pd = np.empty([p2,q])
pdtest = np.empty([p2,q])

h = 0
for i in kidlist1:
    for j in i:
        temp1 = testdata[j]
        temp2 = plotdata[j]
        for k in range(0,q):
            training[h,k] = temp1[k]
            pd[h,k] = temp2[k]
        h = h+1

'''
h = 0
for i in kidlist2:
    for j in i:
        temp1 = testdata2[np.where(dataset2[:,0]==j)[0][0]]
        for k in range(0,q2):
            training[h,k] = temp1[k]
        h = h+1
'''

h = 0
for i in testlist1:
    for j in i:
        temp1 = testdata[j]
        temp2 = plotdata[j]
        for k in range(0,q):
            sampletest[h,k] = temp1[k]
            pdtest[h,k] = temp2[k]
        h = h+1

'''
h = 0
for i in testlist2:
    for j in i:
        temp1 = testdata2[np.where(dataset2[:,0]==j)[0][0]]
        for k in range(0,q2):
            sampletest[h,k] = temp1[k]
        h = h+1
'''

training2 = training[100:]
sample2 = sampletest[100:]
pd2 = pd[100:]
pdtest2 = pdtest[100:]

h = 0
for i in (kidlist1 + testlist1):
    for j in i:
        #print g,h,j
        #if i==quiet or i==qtest:            # col[:,0] entire set
            #col[h] = ['white', 'black', 'white', 'black', 'white']
            #labels[h] = [0, -1, 0, -1, 0]
        if i==rrlyrae or i==rtest:        # col[:,1] variables only
            col[h] = ['magenta', 'magenta', 'magenta', 'magenta', 'green']
            labels[h] = [1, 1, 1, 1, 2]
        elif i==instrip or i==itest:        # col[:,2] combine eclipsing 
            col[h] = ['blue', 'blue', 'blue', 'blue', 'green']
            labels[h] = [3, 3, 3, 3, 2]
        elif i==detached or i==dtest:       # col[:,3] vars only, combine eclipsing
            col[h] = ['cyan', 'cyan', 'green', 'green', 'green']
            labels[h] = [4, 4, 2, 2, 2]
        elif i==semidet or i==sdtest:       # col[:,4] vars versus quiet
            col[h] = ['green', 'green', 'green', 'green', 'green']
            labels[h] = [2, 2, 2, 2, 2]
        elif i==overcontact or i==octest:
            col[h] = ['yellow', 'yellow', 'green', 'green', 'green']
            labels[h] = [5, 5, 2, 2, 2]
        elif i==ellipsoid or i==eltest:
            col[h] = ['orange', 'orange', 'green', 'green', 'green']
            labels[h] = [6, 6, 2, 2, 2]
        elif i==uncertain or i==utest:
            col[h] = ['red', 'red', 'green', 'green', 'green']
            labels[h] = [7, 7, 2, 2, 2]
        h = h+1

#sample = testdata[0:500]
#pdsample = plotdata[0:500]
print np.shape(sampletest)

#fig = pl.figure(1)
#pl.clf()

print "Random Forest"
for i in range(0,2):
    if i==1 or i==3:
        X_train = training
        y_train = labels[:70,i]
        X_test = sampletest
        y_test = labels[70:,i]
    else:
        X_train = training
        y_train = labels[:70,i]
        X_test = sampletest
        y_test = labels[70:,i]

    narray = np.zeros(100)
    for j in range(0,100):
        rfc = RandomForestClassifier(n_estimators=(30), max_depth=(20))
        rfc.fit(X_train, y_train)
        y_pred = rfc.predict(X_test)
            
        for i in range(0,len(y_pred)):
            #if y_pred[i] == y_test[i]:
            if (y_pred[i] == y_test[i]) or ((y_pred[i]==2 or y_pred[i]==4 or y_pred[i]==5 or y_pred[i]==6) and (y_test[i]==2 or y_test[i]==4 or y_test[i]==5 or y_test[i]==6)):
                #print i, y_pred[i], y_test[i]
                narray[j] = narray[j]+1
        #print n*100.0/len(y_test)
    
    print '{:3.0f}/{:3d}, {:2.2%}, n_estimators=30, max_depth=20'.format(np.mean(narray), len(y_test), np.mean(narray)*1.0/len(y_test))

    n=0
    accuracy = np.zeros(100)
    box = np.zeros([6,6])
    for m in range(0,100):
        rfc = RandomForestClassifier(n_estimators=30, max_depth=20)
        y_pred = rfc.fit(X_train, y_train).predict(X_test)
        for i in range(0,len(y_pred)):
            if y_pred[i] == y_test[i]:
                n = n+1
                accuracy[m] = accuracy[m]+1
            if (y_pred[i]==2 or y_pred[i]==4 or y_pred[i]==5 or y_pred[i]==6) and (y_test[i]==2 or y_test[i]==4 or y_test[i]==5 or y_test[i]==6):
                box[1,1] = box[1,1]+1
            elif y_pred[i]==3 and y_test[i]==3:
                box[2,2] = box[2,2]+1
            elif y_pred[i]==3 and (y_test[i]!=3 or y_test[i]!=1):
                box[1,2] = box[1,2]+1
            elif y_test[i]==3 and (y_pred[i]!=3 or y_pred[i]!=1):
                box[2,1] = box[2,1]+1
            elif y_pred[i]==1 and y_test[i]!=1:
                box[1,0] = box[1,0]+1
            elif y_test[i]==1 and y_pred[i]!=1:
                box[0,1] = box[0,1]+1
            else:
                box[y_test[i]-1,y_pred[i]-1] = box[y_test[i]-1,y_pred[i]-1] + 1
    print "Random Forest Classifier, n_estimators=30, max_depth=20: ",np.mean(accuracy)/0.72,np.std(accuracy)/0.72
    for i in range(0,6):
        for j in range(0,6):
            print '{:5.2f} '.format(box[i,j]/100.0),
        print
