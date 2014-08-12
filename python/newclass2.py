# runs multiple classifiers on multiple training sets and test sets
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

n2=len(dataset)
print n2
q=14
testdata = np.empty([n2,q])
plotdata = np.empty([n2,q])

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

'''
quiet = [0]*100
qtest = [0]*100

qui=0
j=0
while 1:
    if (logg[j]>=(6.0-0.0004*teff[j]) or logg[j]>=4.0) and logdata[j,5]<=1 and \
logdata[j,15]<=1 and teff[j]<=6100:
        if qui<100: quiet[qui] = dataset[j,0]
        elif qui<200: qtest[qui-100] = dataset[j,0]
        qui = qui+1
    if qui>=200: break
    j = j+1
'''
p2 = len(rrlyrae + instrip + detached + semidet + overcontact + ellipsoid)
p1 = p2 + 100
kidlist1 = [rrlyrae, instrip, detached, semidet, overcontact, ellipsoid]
testlist1 = [rtest, itest, dtest, sdtest, octest, eltest]

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

h = 0
for i in testlist1:
    for j in i:
        temp1 = testdata[j]
        temp2 = plotdata[j]
        for k in range(0,q):
            sampletest[h,k] = temp1[k]
            pdtest[h,k] = temp2[k]
        h = h+1

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

print "Gaussian Naive Bayes"
gnb = naive_bayes.GaussianNB()
for i in range(0,5):
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

    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    
    n=0
    for i in range(0,len(y_pred)):
        if y_pred[i] == y_test[i]:
            #print i, y_pred[i], y_test[i]
            n = n+1
    print '{:3d}/{:3d}, {:2.2%}'.format(n, len(y_test), n*1.0/len(y_test))

#samplecol = gnb.predict(X_test)
#posterior = gnb.predict_proba(X_test)
#for i in range(0,len(y_test)):
    #if y_test[i] != y_pred[i]:
        #print y_test[i], y_pred[i]
        #print posterior[i]

# multinomial naive bayes requires a non-negative training set

print "Bernoulli Naive Bayes"
bnb = naive_bayes.BernoulliNB()
for i in range(0,5):
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

    bnb.fit(X_train, y_train)
    y_pred = bnb.predict(X_test)
    
    n=0
    for i in range(0,len(y_pred)):
        if y_pred[i] == y_test[i]:
            #print i, y_pred[i], y_test[i]
            n = n+1
    print '{:3d}/{:3d}, {:2.2%}'.format(n, len(y_test), n*1.0/len(y_test))

print "Random Forest"
# changing n_estimators will improve some classes at the expense of others
for k in range(3,4):
    for l in range(4,5):
        for i in range(0,5):
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
                rfc = RandomForestClassifier(n_estimators=(k*10), max_depth=(l*5))
                rfc.fit(X_train, y_train)
                y_pred = rfc.predict(X_test)
            
                for i in range(0,len(y_pred)):
                    if y_pred[i] == y_test[i]:
                #print i, y_pred[i], y_test[i]
                        narray[j] = narray[j]+1
        #print n*100.0/len(y_test)
            print '{:3.0f}/{:3d}, {:2.2%}, n_estimators={:2d}, max_depth={:2d}'.format(np.mean(narray), len(y_test), np.mean(narray)*1.0/len(y_test),(k*10),(l*5))
'''
            narray = np.zeros(100)
            for j in range(0,100):
                rfc = RandomForestClassifier(n_estimators=k*10, max_depth=l*5)
                rfc.fit(X_test, y_test)
                y_pred = rfc.predict(X_train)
            
                for i in range(0,len(y_pred)):
                    if y_pred[i] == y_train[i]:
                #print i, y_pred[i], y_test[i]
                        narray[j] = narray[j]+1
        #print n*100.0/len(y_test)
            print '{:3.0f}/{:3d}, {:2.2%}, n_estimators={:2d}, max_depth={:2d}'.format(np.mean(narray), len(y_test), np.mean(narray)*1.0/len(y_test),k*10,l*5)
'''
print "Support Vector Machines"
# no effect from changing degree
for i in range(0,4):
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

    deg = 0
    ncorrect = 0
    mysvm = svm.SVC()
    mysvm.fit(X_train, y_train)
    y_pred = mysvm.predict(X_test)
    
    n=0
    for i in range(0,len(y_pred)):
        if y_pred[i] == y_test[i]:
            #print i, y_pred[i], y_test[i]
            n = n+1
    print '{:3d}/{:3d}, {:2.2%}'.format(n, len(y_test), n*1.0/len(y_test))

print "Gaussian Mixture Models"
# changing n_components will improve so components at the expense of others
for i in range(0,5):
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

    if i>=2: n=3
    else: n=6

    narray = np.zeros(100)
    for j in range(0,100):
        gmm = GMMBayes(n_components=n)
        gmm.fit(X_train, y_train)
        y_pred = gmm.predict(X_test)
    
        for k in range(0,len(y_pred)):
            if y_pred[k] == y_test[k]:
                #print i, y_pred[i], y_test[i]
                narray[j] = narray[j]+1
        #print n*100.0/len(y_test)
    print '{:3.0f}/{:3d}, {:2.2%}, n_components={:d}'.format(np.mean(narray), len(y_test), np.mean(narray)*1.0/len(y_test), n)

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
'''
ax1 = pl.subplot(221,axisbg='black')
ax1.scatter(pd[:,0],pd[:,1],c=col[:,0],lw=0,s=7,picker=True)
pl.xlabel('T_eff')
pl.ylabel('log(g)')

ax2 = pl.subplot(222,axisbg='black')
ax2.scatter(pd[:,2],pd[:,3],c=col[:,0],lw=0,s=7,picker=True)
pl.xlabel('log-first peak height')
pl.ylabel('log-second peak height')

ax3 = pl.subplot(223,axisbg='black')
ax3.scatter(pd[:,4],pd[:,5],c=col[:,0],lw=0,s=7,picker=True)
pl.xlabel('log-first peak period (d)')
pl.ylabel('log-second peak period (d)')

ax4 = pl.subplot(224,axisbg='black')
ax4.scatter(pd[:,8],pd[:,9],c=col[:,0],lw=0,s=7,picker=True)
pl.xlabel('log-amplitude')
pl.ylabel('RMS deviation')

pl.show()
'''
